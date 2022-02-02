"""
    Compress the video through gradient-based optimization.
"""

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
from PIL import Image
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
from inference import inference, encode
from examine import examine
import pymongo
from munch import *
from utils.seed import set_seed

# from knob.control_knobs import framerate_control, quality_control
sns.set()
set_seed()



# set_trace()
conf.space = munchify(settings.configuration_space.to_dict())
state = {}

len_gt_video = 10
logger = logging.getLogger("diff")


# default_size = (800, 1333)
conf.serialize_order = list(settings.backprop.tunable_config.keys())

# sanity check
for key in settings.backprop.frozen_config.keys():
    assert key not in conf.serialize_order, f'{key} cannot both present in tunable config and frozen config.'



def read_expensive_from_config(gt_args, state, app, db, command_line_args):

    average_video = None
    average_bw = 0
    sum_prob = 0
    
    for args in conf.serialize_most_expensive_state(gt_args.copy(), conf.state2config(state), conf.serialize_order):


        # encode
        # args['gamma'] = 1.0
        video_name, remaining_frames = encode(args)
        video = read_video_to_tensor(video_name)

        
        # video = video * prob

        # sum_prob += prob

        # if average_video is None:
        #     average_video = video
        # else:
        #     average_video = average_video + video

        # update statistics of random choice.
        args.command_line_args = vars(command_line_args)
        args.settings = settings.to_dict()
        stat = examine(args,gt_args,app,db)

        return stat, args, video


        # assert ret.keys() == {}.keys() or stat.keys() == ret.keys()

    #     for key in stat:
    #         if type(stat[key]) in [int, float]:
    #             ret[key] += stat[key] * prob

    #     Path(video_name).unlink()


    # ret.update({'video': average_video})
    # ret = dict(ret)
    # return ret


def optimize(args: dict, key: str, grad: torch.Tensor, gt_config, gt_video):
    
    # assert not hq_video.requires_grad

    

    configs = conf.space[key]
    args = args.copy()

    # set_trace()

    hq_index = configs.index(args[key])
    lq_index = hq_index + 1
    assert lq_index < len(configs)
    delta = 1.0 / (len(configs) - 1)
    x = state[key]
    if x.grad is None:
        x.grad = torch.zeros_like(x)

    def check():

        # logger.info(f'Index: HQ {hq_index} and LQ {lq_index}')

        logger.info(f'Searching {key} between HQ {configs[hq_index]} and LQ {configs[lq_index]}')
        
        args[key] = configs[lq_index]
        lq_name, lq_config = encode(args)
        lq_video = read_video_to_tensor(lq_name)
        # print(lq_config)

        args[key] = configs[hq_index]
        hq_name, hq_config = encode(args)
        hq_video = read_video_to_tensor(hq_name)
        # print(hq_config)

        diff_thresh = settings.backprop.difference_threshold

        hq_diff = (hq_video - gt_video).abs()
        hq_diff = hq_diff.sum(dim=1, keepdim=True)
        logger.info(f'%.5f pixels in HQ are neglected', (hq_diff>diff_thresh).sum()/(hq_diff>-1).sum()) 
        hq_diff[hq_diff > diff_thresh] = 0
        
        lq_diff = (lq_video - gt_video).abs()
        lq_diff = lq_diff.sum(dim=1, keepdim=True)
        logger.info(f'%.5f pixels in LQ are neglected', (lq_diff>diff_thresh).sum()/(lq_diff>-1).sum()) 
        lq_diff[lq_diff > diff_thresh] = 0

        

        if (hq_video - lq_video).abs().mean() > 1e-5: 

            logger.info('Search completed.')
            
            left, right = 1 - delta * hq_index, 1 - delta * lq_index
            # print(f'Left {left}, x {x}, right {right}')
            if not left >= x > right:
                breakpoint()
            assert left >= x > right


            # accuracy gradient term
            # wanna minimize this. So positive gradient.
            lq_ratio = (left - x) / (left - right)
            hq_ratio = (x - right) / (left - right)

            hq_losses = {
                'rec': (grad * hq_diff).abs().mean().item(),
                'bw': hq_config['bw'] / gt_config['bw'],
                'com': len(hq_config['encoded_frames']) / settings.segment_length
            }

            lq_losses = {
                'rec': (grad * lq_diff).abs().mean().item(),
                'bw': lq_config['bw'] / gt_config['bw'],
                'com': len(lq_config['encoded_frames']) / settings.segment_length
            }

            delta_loss = {i: hq_losses[i] - lq_losses[i] for i in hq_losses.keys()}

            logger.info(f'{key}({configs[hq_index]}): {hq_losses}')
            logger.info(f'{key}({configs[lq_index]}): {lq_losses}')
            logger.info(f'Delta: {delta_loss}')
            
            logger.info('HQ loss: %.4f', sum(hq_losses.values()))
            logger.info('LQ loss: %.4f', sum(lq_losses.values()))

            objective = {i: hq_losses[i] * hq_ratio + lq_losses[i] * lq_ratio for i in hq_losses.keys()}
            objective = settings.backprop.reconstruction_loss_weight * objective['rec'] + settings.backprop.bw_weight * objective['bw'] + settings.backprop.compute_weight* objective['com']
            
            
            logger.info(f'Before backwrad, gradient is {x.grad}')

            objective.backward()

            logger.info('Gradient is %.3f', x.grad)
            

            return True

        else:

            return False   


    while (hq_index > 0 or lq_index < len(configs) - 1):

        if check():
            return
        
        hq_index -= 1
        hq_index = max(hq_index, 0)
        lq_index += 1
        lq_index = min(lq_index, len(configs) - 1)
    
    check()








def main(command_line_args):


    # a bunch of initialization.
    
    torch.set_default_tensor_type(torch.FloatTensor)

    db = pymongo.MongoClient("mongodb://localhost:27017/")[settings.collection_name]

    app = DNN_Factory().get_model(settings.backprop.app)

    writer = SummaryWriter(f"runs/{command_line_args.input}/{command_line_args.approach}")

    conf_thresh = settings[app.name].confidence_threshold
    conf_lb = settings[app.name].confidence_lb
    conf_ub = settings[app.name].confidence_ub

    logger.info("Application: %s", app.name)
    logger.info("Input: %s", command_line_args.input)
    logger.info("Approach: %s", command_line_args.approach)
    progress_bar = enlighten.get_manager().counter(
        total=command_line_args.end - command_line_args.start,
        desc=f"{command_line_args.input}",
        unit="10frames",
    )

        
    # initialize configurations pace.
    parameters = []
    for key in conf.serialize_order:
        if key == 'cloudseg':
            parameters.append({
                "params": conf.SR_dnn.net.parameters(), 
                "lr": settings.backprop.tunable_config_lr[key]
            })
            continue
        lr = settings.backprop.lr
        if key in settings.backprop.tunable_config_lr.keys():
            lr = settings.backprop.tunable_config_lr[key]
        state[key] = torch.tensor(settings.backprop.tunable_config[key])
        parameters.append({
            "params": state[key],
            "lr": lr
        })


    # build optimizer
    for tensor in state.values():
        tensor.requires_grad = True
    if settings.backprop.tunable_config.get('cloudseg', None):
        for param in conf.SR_dnn.net.parameters():
            param.requires_grad = True
    optimizer = torch.optim.SGD(parameters)



    for sec in range(command_line_args.start, command_line_args.end):

        progress_bar.update()

        logger.info('\nAt sec %d', sec)

        
        gt_args = munchify(settings.ground_truths_config.to_dict()) 
        gt_args.update({
            'input': command_line_args.input,
            'second': sec,
        })



        # construct average video and average bw
        stat, args, video = read_expensive_from_config(gt_args, state, app, db, command_line_args)
        gt_video_name, gt_video_config = encode(gt_args)
        gt_video = read_video_to_tensor(gt_video_name)
        
        
        if 'gamma' in state:
            video = (video ** state['gamma']).clamp(0, 1)


        # update parameters
        args.command_line_args = vars(command_line_args)
        args.settings = settings.as_dict()
        
        # results = pickle.loads(inference(args, db, app)['inference_result'])
        gt_results = pickle.loads(inference(gt_args, db, app)['inference_result'])
        # for idx, frame in enumerate(tqdm(video)):
        #     with torch.no_grad():
        #         if settings.backprop.tunable_config.cloudseg:
        #             frame = conf.SR_dnn(frame.unsqueeze(0).to('cuda:1')).cpu()
        #         result = app.inference(frame, detach=True, grad=False)
        #         score = torch.sum(result["instances"].scores)
        #         scores[idx] = score
        # scores = {}
        # for idx in results:
        #     scores[key] = torch.sum(results[idx]['instances'].scores)

        # interpolated_fr = conf.state2config(state)['fr']
        # interpolated_fr = interpolated_fr[0][0] * interpolated_fr[0][1] + interpolated_fr[1][0] * interpolated_fr[1][1]

        # set_trace()
        
        # interpolated_fr = ret['#remaining_frames']
        # average_std_score_mean = torch.tensor([scores[i] for i in scores.keys()]).var(unbiased=False).detach()
        # average_sum_score = torch.tensor([scores[i] for i in scores.keys()]).mean()
        # sum_score = torch.tensor([scores[i] for i in scores.keys()]).detach().cpu()



        if settings.backprop.train and sec < 9:

            # take the gradient from the video
            # video.requires_grad = True

            saliencies = {}
            saliencies_tensor = []

            # calculate saliency on each frame.
            if command_line_args.loss_type == 'saliency_error':
                for idx, frame in enumerate(tqdm(video)):
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
                    sum_score = ((score - conf_thresh) * 20).sigmoid().sum()
                    # score_inds = (conf_lb < score) & (score < conf_ub)
                    # logger.info('%d objects need for optimization.' % score_inds.sum())
                    # score = score[score_inds]
                    # sum_score = torch.sum(score)
                    
                    sum_score.backward()

                    saliency = frame_detached.grad.abs().sum(dim=1, keepdim=True)
                    
                    # average across 16x16 neighbors
                    kernel = torch.ones([1, 1, command_line_args.average_window_size, command_line_args.average_window_size])
                    saliency = F.conv2d(saliency, kernel, stride=1, padding=(command_line_args.average_window_size - 1) // 2)
                    saliencies[idx] = saliency
                    saliencies_tensor.append(saliency)


            video.requires_grad_()
            for iteration in range(command_line_args.num_iterations):

                for idx, frame in enumerate(tqdm(video)):

                    if settings.backprop.tunable_config.get('cloudseg', None):
                        frame = conf.SR_dnn(frame.unsqueeze(0).to('cuda:1')).cpu()[0]

                    frame = frame.unsqueeze(0)


                    reconstruction_loss = None
                    result = None
                    gt_frame = gt_video[idx].unsqueeze(0)
                    


                    if command_line_args.loss_type == 'absolute_error':

                        result = app.inference(frame.detach(), detach=True, grad=False)
                        reconstruction_loss =  (gt_frame - frame).abs().mean()
                        reconstruction_loss.backward()

                    # elif command_line_args.loss_type == 'saliency_error_update':

                    #     # # for visualization purpose.
                    #     # result = app.inference(frame.detach(), detach=False, grad=False)

                    #     # calculate saliency again
                    #     frame_detached = frame.detach()
                    #     frame_detached.requires_grad_()
                    #     result = app.inference(frame_detached, detach=False, grad=True)
                    #     # filter out unrelated classes
                    #     result = app.filter_result(result, confidence_check=False)
                    #     score = result["instances"].scores
                    #     score_inds = (conf_lb < score < conf_ub)
                    #     score = score[score_inds]
                    #     sum_score = torch.sum(score)
                        
                    #     sum_score.backward()

                    #     saliency = frame_detached.grad.abs().sum(dim=1, keepdim=True)
                    #     kernel = torch.ones([1, 1, command_line_args.average_window_size, command_line_args.average_window_size])
                    #     saliency = F.conv2d(saliency, kernel, stride=1, padding=(command_line_args.average_window_size - 1) // 2)

                    #     # reconstruction_loss = (saliencies[idx] * (gt_frame - frame).abs()).mean()
                    #     reconstruction_loss = (saliency * (gt_frame - frame).abs()).mean()
                    #     reconstruction_loss.backward()

                    elif command_line_args.loss_type == 'cheat_saliency_error':

                        # # for visualization purpose.
                        # result = app.inference(frame.detach(), detach=False, grad=False)

                        # calculate saliency again
                        frame_detached = frame.detach()
                        frame_detached.requires_grad_()
                        result = app.inference(frame_detached, detach=False, grad=True)
                        # filter out unrelated classes
                        result = app.filter_result(result, confidence_check=False)

                        gt_result = gt_results[idx]
                        gt_ind, res_ind, gt_result, result = app.get_error_confidence_distribution(result, gt_result)

                        in_gt = result[~res_ind]
                        not_in_gt = result[res_ind]

                        FN = in_gt[in_gt.scores < conf_thresh]
                        FP = not_in_gt[not_in_gt.scores > conf_thresh]

                        db['FN_conf'].insert_one({
                            'confidences': FN.scores.tolist(),
                            'command_line_args': vars(command_line_args),
                            'settings': settings.as_dict(),
                            'second': sec,
                            })
                        db['FP_conf'].insert_one({
                            'confidences': FP.scores.tolist(),
                            'command_line_args': vars(command_line_args),
                            'settings': settings.as_dict(),
                            'second': sec
                            })
                        db['Hidden_FN'].insert_one({
                            'count': gt_ind.sum().item(),
                            'command_line_args': vars(command_line_args),
                            'settings': settings.as_dict(),
                            'second': sec
                            })

                        logger.info('FP: %d, FN: %d, Hidden_FN: %d', len(FP), len(FN), gt_ind.sum())

                        (- FP.scores.sum() - (1 - FN.scores).sum()).backward()

                        saliency = frame_detached.grad.abs().sum(dim=1, keepdim=True)
                        kernel = torch.ones([1, 1, command_line_args.average_window_size, command_line_args.average_window_size])
                        saliency = F.conv2d(saliency, kernel, stride=1, padding=(command_line_args.average_window_size - 1) // 2)

                        # reconstruction_loss = (saliencies[idx] * (gt_frame - frame).abs()).mean()
                        reconstruction_loss = (saliency * (gt_frame - frame).abs()).mean()
                        reconstruction_loss.backward()

                        result = {'instances': result}


                    elif command_line_args.loss_type == 'saliency_error':

                        result = app.inference(frame.detach(), detach=True, grad=False)
                        reconstruction_loss = (saliencies[idx] * (gt_frame - frame).abs()).mean()
                        saliency = saliencies[idx]
                        reconstruction_loss.backward()

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

                        



                    writer.add_scalar('Reconstruction/%d' % idx, reconstruction_loss.item(), iteration)

                    # if settings.backprop.early_optimize:
                    #     # logger.info('Early optimize')
                    #     optimizer.step()
                    #     optimizer.zero_grad()


                    if idx % 3 == 0 and settings.backprop.visualize:
                        with torch.no_grad():
                            for key in result['instances'].get_fields():
                                if key == 'pred_boxes':
                                    result['instances'].get(key).tensor = result['instances'].get(key).tensor.detach().cpu()
                                else:
                                    result['instances'].set(key, result['instances'].get(key).detach().cpu())
                            gt_ind, res_ind, gt_filtered, res_filtered = app.get_undetected_ground_truth_index(result, gt_results[idx])


                            image = T.ToPILImage()(frame.detach()[0].clamp(0, 1))
                            image_FN = app.visualize(image, {'instances': gt_filtered[gt_ind]})
                            image_FP = app.visualize(image, {'instances': res_filtered[res_ind]})
                            visualize_heat_by_summarywriter(image_FN, saliency[0][0], f'FN/{idx}', writer, iteration, tile=False)
                            visualize_heat_by_summarywriter(image_FP, saliency[0][0], f'FP/{idx}', writer, iteration, tile=False)

                        if iteration == 0:
                            writer.add_image('GT/%' % idx, gt_video[idx], idx)


                # saliencies_tensor = torch.cat(saliencies_tensor)

                for key in settings.backprop.tunable_config.keys():
                    if key == 'cloudseg':
                        # already optimized when backwarding.
                        continue
                    optimize(args, key, torch.cat(saliencies_tensor), gt_video_config, gt_video)


                # if not settings.backprop.early_optimize:
                optimizer.step()
                optimizer.zero_grad()

                # truncate                
                for tensor in state.values():
                    tensor.requires_grad = False
                for key in conf.serialize_order:
                    if key == 'cloudseg':
                        continue
                    if state[key] > 1.:
                        state[key][()] = 1.
                    if state[key] < 1e-7:
                        state[key][()] = 1e-7
                for tensor in state.values():
                    tensor.requires_grad = True

                # print out current state
                state_str = ""
                for key in conf.serialize_order:
                    if key == 'cloudseg':
                        param = list(conf.SR_dnn.net.parameters())[0]
                        logger.info('CloudSeg: gradient of layer 0 mean: %.3f, std: %.3f', param.grad.mean(), param.grad.std())
                        continue
                    state_str += '%s : %.3f, grad: %.7f\n' % (key, state[key], state[key].grad)

                logger.info(f'Current state: {state}')


                

        # if args.train:
        #     (-(1/len_gt_video) * last_score).backward(retain_graph=True)
        # average_sum_score = average_sum_score +  (-(1/len_gt_video) * last_score).item()
            



        # objective = (settings.backprop.sum_score_mean_weight * average_sum_score + settings.backprop.std_score_mean_weight * average_std_score_mean)
        # true_obj = (settings.backprop.sum_score_mean_weight * true_average_score + settings.backprop.std_score_mean_weight * true_average_std_score_mean  - settings.backprop.compute_weight * interpolated_fr.detach().item())
        
        
        
        # # logger.info('QP: %.3f, Res: %.3f, Fr: %.3f', state['qp'], state['res'], state['fr'])
        # # logger.info('qpgrad: %.3f, frgrad: %.3f, resgrad: %.3f', state['qp'].grad, state['fr'].grad, state['res'].grad)
        
        # # logger.info('Score: %.3f, std: %.3f, bw : %.3f, Obj: %.3f', average_sum_score, average_std_score_mean, ret['bw'], objective.item())
        # logger.info('Reconstruction loss: %.3f', reconstruction_loss.item())

        # performance = examine(args, gt_args, app, db)
        # logger.info('F1: %.3f', performance['f1'])

        # logger.info('True : %.3f, Tru: %.3f, Tru: %.3f, Tru: %.3f', true_average_score, true_average_std_score_mean, true_average_bw, true_obj)

        # truncate            
        # logger.info(f'Current config: {conf.state2config(state)}')

        

        # choose = conf.random_serialize(video_name, conf.state2config(state))

    mean = torch.tensor(means).mean().item()

    logger.info('Overall mean quality: %.3f', mean)

    # with open('config.yaml', 'a') as f:
    #     f.write(yaml.dump([{
    #         '#frames': fid,
    #         'bw': mean * Path(args.hq).stat().st_size + (1-mean) * Path(args.lq).stat().st_size,
    #         'video_name': args.output
    #     }]))

    # with open('diff.yaml', 'a') as f:
    #     f.write(yaml.dump({
    #         'acc': accs,
    #         'compute': computes,
    #         'size': sizes
    #     }))

    # print(torch.tensor(accs).mean() + torch.tensor(computes).mean() + torch.tensor(sizes).mean())
        


if __name__ == "__main__":

    # set the format of the logger
    coloredlogs.install(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s",
        datefmt="%H:%M:%S",
        level="INFO",
    )

    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--freq",
    #     help="The video file names. The largest video file will be the ground truth.",
    #     default=1,
    #     type=int
    # )

    # parser.add_argument(
    #     '--lr',
    #     help='The learning rate',
    #     type=float,
    #     default=0.003
    # )

    # parser.add_argument(
    #     '--qp',
    #     help='The quantization parameter',
    #     type=float,
    #     default=1.
    # )

    # parser.add_argument(
    #     '--fr',
    #     help='The frame rate',
    #     type=float,
    #     default=1.
    # )

    # parser.add_argument(
    #     '--res',
    #     help='The resolution',
    #     type=float,
    #     default=1.
    # )

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
    # parser.add_argument(
    #     '--gamma',
    #     type=float,
    #     help='Adjust the luminance.',
    #     default=1.5,
    # )

    args = parser.parse_args()

    main(args)
