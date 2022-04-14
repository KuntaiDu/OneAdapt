"""
    Compress the video through gradient-based optimization.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import gc
import logging
import time
from pathlib import Path
from typing import Tuple

import coloredlogs
import enlighten
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import pickle
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from torch.utils.tensorboard import SummaryWriter
from utils.visualize_utils import visualize_heat_by_summarywriter
from torchvision import io
from datetime import datetime
import random
import torchvision

import yaml
from config import settings

from pdb import set_trace

from dnn.dnn_factory import DNN_Factory
from dnn.dnn import DNN
# from utils.results import write_results
from utils.video_reader import read_video, read_video_config, read_video_to_tensor
import utils.config_utils as conf
from collections import defaultdict
from tqdm import tqdm
from utils.inference import inference, examine
from utils.encode import encode, tile_mask
from utils.reducto import reducto_process, reducto_process_on_frame, reducto_update_mean_std, reducto_feature2meanstd
import pymongo
from munch import *
from utils.seed import set_seed
from utils.gradient import grad_macroblocks, grad, grad_reducto_expensive, grad_reducto_cheap
from utils.tqdm_handler import TqdmLoggingHandler
from utils.video_visualizer import VideoVisualizer
from detectron2.structures.instances import Instances

import utils.video_visualizer as vis

sns.set()
set_seed()


conf.space = munchify(settings.configuration_space.to_dict())
state = munchify({})

len_gt_video = 10
logger = logging.getLogger("Server")

conf.serialize_order = list(settings.backprop.tunable_config.keys())

# sanity check
for key in settings.backprop.frozen_config.keys():
    assert key not in conf.serialize_order, f'{key} cannot both present in tunable config and frozen config.'



def read_expensive_from_config(gt_args, state, app, db, command_line_args, train_flag=False):

    average_video = None
    average_bw = 0
    sum_prob = 0
    
    for args in conf.serialize_most_expensive_state(gt_args.copy(), conf.state2config(state), conf.serialize_order):
        
        args = args.copy()
        # use ground truth here. For debugging purpose.
        del args.tag
        for key in list(args.keys()):
            if 'qp' in key and 'macroblocks' in state:
                del args.qp
            if 'reducto' in key:
                if train_flag and settings.backprop.reducto_expensive_optimize:
                    # remove frame filtering for training purpose.
                    del args[key]
                else:
                    # this is reducto encoding.
                    args.tag = 'reducto'
        if 'tag' not in list(args.keys()):
            # this is normal encoding.
            args.tag = 'mpeg'

        if 'macroblocks' in state:
            args.macroblocks = state['macroblocks']
            
        # encode
        args.command_line_args = vars(command_line_args)
        args.settings = settings.to_dict()
        video_name, video_config = encode(args)
        logger.info('Ours bandwidth: %d', video_config['bw'])
        video = read_video_to_tensor(video_name)

        # update statistics of random choice.
        inference(args, db, app, video_name, video_config)
        stat = examine(args,gt_args,app,db)

        return stat, args, video




def main(command_line_args):
    
    # a bunch of initialization.
    
    torch.set_default_tensor_type(torch.FloatTensor)

    db = pymongo.MongoClient("mongodb://localhost:27017/")[settings.collection_name]

    app = DNN_Factory().get_model(settings.backprop.app)

    conf_thresh = settings[app.name].confidence_threshold

    logger.info("Application: %s", app.name)
    logger.info("Input: %s", command_line_args.input)
    logger.info("Approach: %s", command_line_args.approach)


    # visualizer
    results_vis = VideoVisualizer(f'debug/{command_line_args.approach}_results.mp4')
    errors_vis = VideoVisualizer(f'debug/{command_line_args.approach}_errors.mp4')

    # initialize optimizer
    placeholder = torch.tensor([1.])
    placeholder.requires_grad = True
    placeholder.grad = torch.zeros_like(placeholder)
    # avoid no parameter to optimize error
    parameters = [{"params": placeholder, "lr": 0.}]
    for key in conf.serialize_order:
        if key == 'cloudseg':
            for param in conf.SR_dnn.net.parameters():
                param.requires_grad = True
            parameters.append({
                "params": conf.SR_dnn.net.parameters(), 
                "lr": settings.backprop.tunable_config_lr[key]
            })
            continue
        elif key == 'macroblocks':
            # will directly solve the closed-form solution. No lr needed.
            state[key] = torch.ones(settings.backprop.macroblock_shape).int() * settings.backprop.qps[0]
            continue
        elif 'reducto' in key:
            # will directly solve the closed-form solution. No lr needed.
            state[key] = torch.tensor(settings.backprop.tunable_config[key])
            continue
        
        lr = settings.backprop.lr
        if key in settings.backprop.tunable_config_lr.keys():
            lr = settings.backprop.tunable_config_lr[key]
        
        state[key] = torch.tensor(settings.backprop.tunable_config[key])
        state[key].requires_grad=True
        parameters.append({
            "params": state[key],
            "lr": lr
        })
    
    # build optimizer
    optimizer = torch.optim.Adam(parameters, betas=(0.5, 0.5))



    for sec in tqdm(range(command_line_args.start, command_line_args.end), unit='sec', desc='progress'):

        # the text for visualization
        vis.text = ""
        
        # get the ground truth configuration and the camera-side ground truth video
        gt_args = munchify(settings.ground_truths_config.to_dict()) 
        gt_args.update({
            'input': command_line_args.input,
            'second': sec,
        })
        gt_video_name, gt_video_config = encode(gt_args)
        gt_video = read_video_to_tensor(gt_video_name)
        logger.info('Ground truth bandwidth: %d', gt_video_config['bw'])
        # update the mean and the std of reducto thresholds.
        for key in state.keys():
            if 'reducto' in key:
                reducto_update_mean_std(gt_video, state, gt_args, db)
                break
            
            
        # encode and inference on the current video.
        train_flag = (settings.backprop.train and (sec-command_line_args.start) % command_line_args.frequency == 0)
        stat, args, server_video = read_expensive_from_config(gt_args, state, app, db, command_line_args, train_flag)
        my_video_config = stat['my_video_config']
        logger.info('Actual compute: %d' % my_video_config['compute'])
        vis.text += ("Comp: %d\n"
                          "Acc : %.3f\n"
                          "Bw  : %.3f\n") % (my_video_config['compute'], stat['acc'], stat['norm_bw'])
        gt_results = pickle.loads(stat['gt_inference_result'])
        my_results = pickle.loads(stat['my_inference_result'])         


        if train_flag:

            saliencies = {}
            raw_saliencies_tensor = []
            saliencies_tensor = []
            inference_results = {}
            camera_video = server_video
            compute = None
            
            if any('reducto' in key for key in state.keys()) and not settings.backprop.reducto_expensive_optimize:
                # get the differentiable reducto processed video on the camera
                camera_video, metrics = reducto_process(gt_video, state)
                # new_vis.text = 'Estimated compute: %.3f' % metrics['compute'].item()
                # vis.text += new_vis.text
                # logger.info(new_vis.text)

            # calculate saliency on each frame.
            if command_line_args.loss_type == 'saliency_error':
                for idx, frame in enumerate(tqdm(server_video, unit='frame', desc='saliency')):
                    gt_frame = gt_video[idx]
                    if settings.backprop.tunable_config.get('cloudseg', None):
                        with torch.no_grad():
                            frame = conf.SR_dnn(frame.unsqueeze(0).to('cuda:1')).cpu()[0]
                    frame = frame.unsqueeze(0)

                    frame_detached = frame.detach()
                    frame_detached.requires_grad_()
                    result = app.inference(frame_detached, detach=False, grad=True)
                    # filter out unrelated classes
                    result = app.filter_result(result, confidence_check=False)
                    score = result["instances"].scores
                    if settings.backprop.saliency_type == 'sigmoid':
                        sum_score = ((score - conf_thresh) * 20).sigmoid().sum()
                    elif settings.backprop.saliency_type == 'sum':
                        sum_score = score[score > conf_thresh].sum()
                    # score_inds = (conf_lb < score) & (score < conf_ub)
                    # logger.info('%d objects need for optimization.' % score_inds.sum())
                    # score = score[score_inds]
                    # sum_score = torch.sum(score)
                    inference_results[idx] = result                    
                    sum_score.backward()

                    saliency = frame_detached.grad
                    
                    raw_saliencies_tensor.append(saliency.clone())
                    saliency = saliency.abs().sum(dim=1, keepdim=True)
                    
                    # average across 16x16 neighbors
                    kernel = torch.ones([1, 1, command_line_args.average_window_size, command_line_args.average_window_size])
                    saliency = F.conv2d(saliency, kernel, stride=1, padding=(command_line_args.average_window_size - 1) // 2)
                    saliencies[idx] = saliency
                    saliencies_tensor.append(saliency)
                    


            # video.requires_grad_()
            for iteration in range(command_line_args.num_iterations):

                for idx, frame in enumerate(tqdm(server_video, unit='frame', desc='optimize')):

                    fid = sec * 10 + idx

                    if settings.backprop.tunable_config.get('cloudseg', None):
                        frame = conf.SR_dnn(frame.unsqueeze(0).to('cuda:1')).cpu()[0]

                    frame = frame.unsqueeze(0)


                    reconstruction_loss = None
                    result = None
                    



                    if command_line_args.loss_type == 'saliency_error':
                        

                        # result = app.inference(frame.detach(), detach=True, grad=False)
                        reconstruction_loss = (saliencies[idx] * (gt_video[idx].unsqueeze(0) - camera_video[idx].unsqueeze(0)).abs()).mean()
                        if frame.grad_fn is not None:
                            # frame is generated by differentiable knobs. Backward.
                            reconstruction_loss.backward(retain_graph=True)

                    elif command_line_args.loss_type == 'feature_error':

                        gt_result = app.inference(gt_frame, detach=False, grad=False, feature=True)
                        frame.retain_grad()
                        result = app.inference(frame, detach=False, grad=True, feature=True)

                        feature_diffs = []
                        for i in range(5):
                            feature_diffs.append((gt_result['features'][i] - result['features'][i]).abs().mean())

                        reconstruction_loss = sum(feature_diffs)
                        reconstruction_loss.backward()
                        del result['features']
                        del gt_result['features']

                        saliency = frame.grad.abs().sum(dim=1, keepdim=True)
                        kernel = torch.ones([1, 1, command_line_args.average_window_size, command_line_args.average_window_size])
                        saliency = F.conv2d(saliency, kernel, stride=1, padding=(command_line_args.average_window_size - 1) // 2)

                    else:
                        raise NotImplementedError
                        
                # backward cost parameters
                if 'metrics' in locals():
                    if metrics['compute'].grad_fn is not None:
                        (settings.backprop.compute_weight * metrics['compute']).backward()


                reducto_optimized = False
                for key in settings.backprop.tunable_config.keys():
                    if key == 'cloudseg':
                        # already optimized when backwarding.
                        continue
                    elif key == 'macroblocks':
                        grad_macroblocks(args, key, torch.cat(raw_saliencies_tensor), gt_video_config, gt_video)
                        continue
                    elif 'reducto' in key:
                        if not reducto_optimized:
                            if settings.backprop.reducto_expensive_optimize:
                                grad_reducto_expensive(state, gt_video, inference_results, app)
                            else:
                                grad_reducto_cheap(state, gt_video, torch.cat(saliencies_tensor))
                                
                            reducto_optimized = True
                            
                        continue
                    grad(args, key, torch.cat(saliencies_tensor), gt_video_config, gt_video)

                optimizer.step()
                optimizer.zero_grad()

                # round to [0,1]         
                for key in conf.serialize_order:
                    if key == 'cloudseg':
                        continue
                    if key == 'macroblocks':
                        continue
                    if 'reducto' in key:
                        continue
                    state[key].requires_grad = False
                    if state[key] > 1.:
                        state[key][()] = 1.
                    if state[key] < 1e-7:
                        state[key][()] = 1e-7
                    state[key].requires_grad = True

                # print out current state
                if not settings.backprop.reducto_expensive_optimize:
                    state_str = ""
                    for key in conf.serialize_order:
                        if key == 'cloudseg':
                            param = list(conf.SR_dnn.net.parameters())[0]
                            logger.info('CloudSeg: gradient of layer 0 mean: %.3f, std: %.3f', param.grad.mean(), param.grad.std())
                            continue
                        if key == 'macroblocks':
                            param = state[key]
                            logger.info('Macroblocks: mean: %.3f', param.float().mean())
                            continue
                        if 'reducto' in key:
                            param = state[key]
                            logger.info(f'{key}: %.3f', param.float().mean())
                            continue
                        state_str += '%s : %.3f, grad: %.7f\n' % (key, state[key], state[key].grad)

                    logger.debug(f'Current state: {state}')
                    
        
        # visualize
        for idx, frame in enumerate(tqdm(server_video, desc='visualize', unit='frame')):

            image = T.ToPILImage()(frame.clamp(0, 1))
            

            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 24)
            # manually bold the text :-P
            draw.multiline_text((10, 10), vis.text, fill=(255, 0, 0), font=font)
            draw.multiline_text((11, 10), vis.text, fill=(255, 0, 0), font=font)
            draw.multiline_text((12, 10), vis.text, fill=(255, 0, 0), font=font)

            my_result = app.filter_result(my_results[idx])
            gt_result = app.filter_result(gt_results[idx], gt=True)

            gt_ind, my_ind, gt_filtered, my_filtered = app.get_undetected_ground_truth_index(my_result, gt_result)

            image_error = app.visualize(image, {"instances": Instances.cat([gt_filtered[gt_ind], my_filtered[my_ind]])})
            image_inference = app.visualize(image, my_result)
            errors_vis.add_frame(image_error)
            results_vis.add_frame(image_inference)
            
        logger.info('Visualize text:\n%s', vis.text)
        


if __name__ == "__main__":

    # set the format of the logger
    formatter = coloredlogs.ColoredFormatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s",
        datefmt="%H:%M:%S",
    )

    handler = TqdmLoggingHandler()
    handler.setFormatter(formatter)
    logging.basicConfig(
        handlers=[handler],
        level='INFO'
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--loss_type',
        type=str,
        required=True
    )

    parser.add_argument(
        '-i',
        '--input',
        help='The format of input video.',
        type=str,
        required=True
    )

    parser.add_argument(
        '--start',
        help='The total secs of the video.',
        required=True,
        type=int
    )

    parser.add_argument(
        '--end',
        help='The total secs of the video.',
        required=True,
        type=int
    )

    parser.add_argument(
        '--num_iterations',
        help='The total secs of the video.',
        required=True,
        type=int
    )

    parser.add_argument(
        '--frequency',
        help='The total secs of the video.',
        required=True,
        type=int
    )


    parser.add_argument(
        "--app", 
        type=str, 
        help="The name of the model.", 
        default='EfficientDet-d2',
    )

    parser.add_argument(
        "--average_window_size",
        type=int,
        help='The window size for saliency averaging',
        default=17
    )

    parser.add_argument(
        '--approach',
        type=str,
        required=True
    )

    args = parser.parse_args()

    main(args)
