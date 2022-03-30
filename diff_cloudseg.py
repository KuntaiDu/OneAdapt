"""
    Compress the video through gradient-based optimization.
"""

import argparse
import gc
import logging
import time
from pathlib import Path
from typing import Tuple
from matplotlib.colors import ListedColormap
from detectron2.structures.boxes import pairwise_iou
import cv2 as cv
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
import utils.bbox_utils as bu
from config import settings

from pdb import set_trace

from dnn.dnn_factory import DNN_Factory
from dnn.dnn import DNN
# from utils.results import write_results
from utils.video_reader import read_video, read_video_config
import utils.config_utils as conf
from utils.mask_utils import generate_mask_from_regions
from collections import defaultdict
from tqdm import tqdm
from inference import inference, encode
from examine import examine
import pymongo
from munch import *
import numpy as np
import os
# from knob.control_knobs import framerate_control, quality_control
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #TODO: Do we need deterministic in cudnn ? Double check
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print ("Seeded everything")




def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat([(boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2]], 1)  # w, h


sns.set()


# set_trace()
conf.space = munchify(settings.configuration_space.to_dict())
state = {}

len_gt_video = 10
logger = logging.getLogger("diff")



# default_size = (800, 1333)
conf.serialize_order = list(settings.backprop.tunable_config.keys())
def plot_cdf(scores, sec, fid, method):
    sorted_vals = np.sort(scores)
    p = 1. * np.arange(len(sorted_vals))/(len(sorted_vals) - 1)

    fig, ax = plt.subplots(1, 1, figsize=(11, 5), dpi=200)
    ax.plot(sorted_vals, p)
    fig.savefig(f"plots/cdf_{sec}_{fid}_{method}.png")

def plot_pdf(scores, sec, fid, method):

    fig, ax = plt.subplots(1, 1, figsize=(11, 5), dpi=200)
    count, bins_count = np.histogram(scores, bins=100)
    pdf = count / sum(count)
    plt.plot(bins_count[1:], pdf)
    fig.savefig(f"plots/cdf_{sec}_{fid}_{method}.png")

def quantize_confidence_score(scores, confidence_score_dict):
    round_scores = [round(i.cpu().detach().item(), 4) for i in scores]
    for round_score in round_scores:
        if str(round_score) not in confidence_score_dict:
            confidence_score_dict[str(round_score)] = 0
        else:
            confidence_score_dict[str(round_score)] += 1


def augment(result, lengt):

    factor = (lengt + (len(result) - 1)) // len(result)

    return torch.cat([result[i // factor][None, :, :, :] for i in range(lengt)])




def read_expensive_from_config(gt_args: Munch, state, app: DNN, db: pymongo.database.Database) -> Tuple[dict, Munch]:

    average_video = None
    average_bw = 0
    sum_prob = 0

    # ret = defaultdict(lambda: 0)

    for args in conf.serialize_most_expensive_state(gt_args.copy(), conf.state2config(state), conf.serialize_order):

        print(conf.state2config(state))
        # encode
        # args['gamma'] = 1.0
        video_name, remaining_frames = encode(args)
        video = list(read_video(video_name))
        video = torch.cat([i[1] for i in video])
        # video = F.interpolate(video, size=default_size)
        video = augment(video, len_gt_video)


        # video = video * prob

        # sum_prob += prob

        # if average_video is None:
        #     average_video = video
        # else:
        #     average_video = average_video + video

        # update statistics of random choice.
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


def optimize(args: dict, key: str, grad: torch.Tensor):

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
        lq_name, lq_remaining_frames = encode(args)
        lq_video = torch.cat([i[1] for i in list(read_video(lq_name))])
        print(lq_remaining_frames)

        args[key] = configs[hq_index]
        hq_name, hq_remaining_frames = encode(args)
        hq_video = torch.cat([i[1] for i in list(read_video(hq_name))])
        print(hq_remaining_frames)



        if (hq_video - lq_video).abs().mean() > 1e-5:

            logger.info('Search completed.')

            left, right = 1 - delta * hq_index, 1 - delta * lq_index
            assert left >= x > right



            x.grad += ( ((hq_video - lq_video) / (left - right)) * grad ).sum()

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

def tile_mask(mask, tile_size):
    """
        Here the mask is of shape [1, 1, H, W]
        Eg:
        mask = [    1   2
                    3   4]
        tile_mask(mask, 2) will return
        ret =  [    1   1   2   2
                    1   1   2   2
                    3   3   4   4
                    3   3   4   4]
        This function controlls the granularity of the mask.
    """
    mask = mask[0, 0, :, :]
    t = tile_size
    mask = mask.unsqueeze(1).repeat(1, t, 1).view(-1, mask.shape[1])
    mask = mask.transpose(0, 1)
    mask = mask.unsqueeze(1).repeat(1, t, 1).view(-1, mask.shape[1])
    mask = mask.transpose(0, 1)
    return torch.cat(3 * [mask[None, None, :, :]], 1)

def save_img(gt_frame, file_name):
    numpy_save = np.zeros((gt_frame.shape[1], gt_frame.shape[2], 3))
    img_tensor = T.ToPILImage()(gt_frame)
    img_tensor.save(file_name)
    # numpy_save[:,:,0] = gt_frame.numpy()[0, :,:]
    # numpy_save[:,:,1] = gt_frame.numpy()[1, :,:]
    # numpy_save[:,:,2] = gt_frame.numpy()[2, :,:]
    # cv.imwrite(file_name, numpy_save,
    #                    [cv.IMWRITE_PNG_COMPRESSION, 0])# data.save('gfg_dummy_pic2.png')

def compress_saliency(saliency):
    # saliency_np = saliency.numpy()
    # y = torch.zeros(saliency.shape)
    # print("Fraction: ", np.sum(saliency.numpy() > 2.6)/(saliency.shape[0] * saliency.shape[1]))
    kernel = torch.ones([1, 1, 16, 16])
    saliency_map = F.conv2d(saliency, kernel, stride=16, padding=0)
    flatten_length = len(torch.flatten(saliency_map))
    print(torch.sort(torch.flatten(saliency_map), descending=True)[0][int(flatten_length * 0.1)])
    thres = torch.sort(torch.flatten(saliency_map), descending=True)[0][int(flatten_length * 0.1)]
    saliency_tensor = torch.where(saliency_map >= thres, torch.ones(saliency_map.shape) * 10, torch.ones(saliency_map.shape))
    print("Fraction: ", torch.sum(saliency_tensor[0][0] == 10)/(saliency_map.shape[2] * saliency_map.shape[3]))
    saliency_tensor = tile_mask(saliency_tensor, 16)

    return saliency_tensor

def compress_img(saliency, frame):
    # saliency_np = saliency.numpy()
    # y = torch.zeros(saliency.shape)
    # print("Fraction: ", np.sum(saliency.numpy() > 2.6)/(saliency.shape[0] * saliency.shape[1]))
    kernel = torch.ones([1, 1, 16, 16])
    saliency_map = F.conv2d(saliency, kernel, stride=16, padding=0)

    saliency_tensor = torch.where(saliency_map / 256 > 10 , torch.ones(saliency_map.shape), torch.zeros(saliency_map.shape))
    print("Fraction: ", torch.sum(saliency_tensor))
    saliency_tensor = tile_mask(saliency_tensor, 16)
    # print(saliency_tensor.shape)
    # print(frame.shape)
    frame[0, :,:] = frame[0, :,:] * saliency_tensor[0][0]
    frame[1, :,:] = frame[1, :,:] * saliency_tensor[0][0]
    frame[2, :,:] = frame[2, :,:] * saliency_tensor[0][0]
    return frame

def main(command_line_args):

    cmapmine = ListedColormap(['b', 'w'], N=2)

    # a bunch of initialization.
    set_seed(0)
    torch.set_default_tensor_type(torch.FloatTensor)

    db = pymongo.MongoClient("mongodb://localhost:27017/")[settings.collection_name]

    app = DNN_Factory().get_model(settings.backprop.app)
    print(app)

    writer = SummaryWriter(f"runs/{command_line_args.approach}")
    conf_thresh = settings[settings.backprop.app].confidence_threshold
    conf_lb = settings[settings.backprop.app].confidence_lb
    conf_ub = settings[settings.backprop.app].confidence_ub

    logger.info("Application: %s", app.name)
    logger.info("Input: %s", command_line_args.input)
    logger.info("Approach: %s", command_line_args.approach)
    progress_bar = enlighten.get_manager().counter(
        total=command_line_args.end - command_line_args.start,
        desc=f"{command_line_args.input}",
        unit="10frames",
    )

    conf.SR_dnn.net.train()
    # initialize configurations pace.
    parameters = []
    for key in settings.backprop.tunable_config.keys():
        if key == 'cloudseg':
            parameters.append({
                "params": conf.SR_dnn.net.parameters(),
                "lr": settings.backprop.tunable_config_lr[key]
            })
            continue
        state[key] = torch.tensor(settings.backprop.tunable_config[key])
        parameters.append({
            "params": state[key],
            "lr": settings.backprop.tunable_config_lr[key]
        })


    # build optimizer
    for tensor in state.values():
        tensor.requires_grad = True


    optimizer = torch.optim.Adam(parameters, lr=settings.backprop.lr)
    all_fn_scores = []
    all_fp_scores = []
    all_scores = []
    confidence_score_dict = {}

    iou_scores = []
    for sec in range(command_line_args.start, command_line_args.end):

        progress_bar.update()

        logger.info('\nAt sec %d', sec)


        gt_args = munchify(settings.ground_truths_config.to_dict())
        gt_args.update({
            'input': command_line_args.input,
            'second': sec,
        })



        # construct average video and average bw
        ret, args, video = read_expensive_from_config(gt_args, state, app, db)
        gt_video_name, _ = encode(gt_args)
        gt_video = list(read_video(gt_video_name))
        gt_video = torch.cat([i[1] for i in gt_video])

        gt_args_new = gt_args.copy()
        gt_args_new['qp'] = 34
        # gt_args_new['res'] = '640x360'
        gt_args_new['name'] = 'qp'
        gt_video_name_new, _ = encode(gt_args_new)
        gt_video_new = list(read_video(gt_video_name_new))
        gt_video_new = torch.cat([i[1] for i in gt_video_new])

        if 'gamma' in state:
            video = (video ** state['gamma']).clamp(0, 1)

        # update parameters
        args.cloudseg = True
        args.command_line_args = vars(command_line_args)
        args.settings = settings.as_dict()

        mask_shape = [
            1,
            1,
            720,
            1280,
        ]
        mask_slice = torch.zeros(mask_shape).float()

        gt, _ = inference(gt_args, db, app)
        # results = pickle.loads(inference(args, db, app)['inference_result'])
        gt_results = pickle.loads(gt['inference_result'])

        if settings.backprop.train and sec % command_line_args.frequency == 0:
            # new_state_dict = torch.load(f'/data/yuhanl/CARN-pytorch/checkpoint/carn_m.pth')
            # conf.SR_dnn.net.load_state_dict(new_state_dict)
            if settings.backprop.tunable_config.cloudseg:
                for param in conf.SR_dnn.net.parameters():
                    param.requires_grad = True
            conf.SR_dnn.net.train()

            # take the gradient from the video
            # video.requires_grad = True
            # if args.second > 0:
            #     new_state_dict = torch.load(f'/data/yuhanl/CARN-pytorch/checkpoint/carn.pth')
            #     conf.SR_dnn.net.load_state_dict(new_state_dict)
            saliencies = {}

            # calculate saliency on each frame.
            # if command_line_args.loss_type == 'feature_error':
            #     for idx, frame in enumerate(tqdm(video)):
            #         gt_frame = gt_video[idx]
            #         if idx > 0: continue
            #         save_img(frame, 'lq_frame.png')
            #         print(frame.shape)
            #         save_img(gt_frame, 'gt_orig.png')
            if command_line_args.loss_type == 'saliency_error' or command_line_args.loss_type == 'saliency_error_plus':
                for idx, frame in enumerate(tqdm(video)):

                    gt_frame = gt_video_new[idx]

                    if settings.backprop.tunable_config.cloudseg:
                        frame = conf.SR_dnn(frame.unsqueeze(0).to('cuda:1')).cpu()

                    frame_detached = frame.detach()
                    frame_detached.requires_grad_()

                    result = app.inference(frame_detached, detach=False, grad=True, raw=True)
                    probs=F.softmax(result)
                    probs = probs[:, 1:, :, :]
                    # print(probs[0])

                    # filter out unrelated classes
                    sum_score = probs.sum()
                    sum_score.backward()
                    # saliency = frame_detached.grad
                    saliency = frame_detached.grad.abs()

                    # # average across 16x16 neighbors
                    # kernel = torch.ones([1, 3, command_line_args.average_window_size, command_line_args.average_window_size])
                    # saliency = F.conv2d(saliency, kernel, stride=1, padding=(command_line_args.average_window_size - 1) // 2)
                    image = T.ToPILImage()(frame.detach()[0].clamp(0, 1))
                    # print(result['instances'].pred_masks)
                    # result['instances'].pred_masks = result['instances'].pred_masks.cpu()
                    # result['instances'].pred_boxes.tensor = result['instances'].pred_boxes.tensor.cpu()

                    # compressed_image = compress_img(saliency, gt_frame)
                    # if os.path.exists(f'images/{sec}') is False:
                    #     os.makedirs(f'images/{sec}')
                    # visualize_heat_by_summarywriter(image, saliency[0][0], f'images/{sec}/{command_line_args.loss_type}_{sec}_{idx}', writer, 0, tile=False)
                    saliencies[idx] = saliency
            if command_line_args.loss_type == 'normal_update_saliency_plus':
                conf.SR_dnn.net.train()
                for idx, frame in enumerate(tqdm(video)):
                    gt_frame = gt_video_new[idx].unsqueeze(0)
                    if settings.backprop.tunable_config.cloudseg:
                        frame = conf.SR_dnn(frame.unsqueeze(0).to('cuda:1')).cpu()
                    frame_detached = frame.detach()
                    frame_detached.requires_grad_()
                    result = app.inference(frame_detached, detach=False, grad=True, raw=True)

                    gt_result = app.inference(gt_frame, detach=False, grad=False, raw=False)
                    gt_result = app.transform_output(gt_result)
                    # print("Result shape: ", result.shape)
                    # print("GT Result shape: ", gt_result.shape)
                    dice_loss = app.dice_loss(result, gt_result) + command_line_args.alpha * torch.norm(frame.detach().cpu()-gt_frame, p=2)#torch.norm(frame-gt_frame, p=2) #- app.dice_loss(result, gt_result)
                    # for param in conf.SR_dnn.net.parameters():
                    #     param.requires_grad = False
                    dice_loss.backward()
                    #
                    # saliency =  frame_detached.grad
                    saliency =  frame_detached.grad.abs()
                    print("Original saliency: ", saliency.mean())
                    saliencies[idx] = saliency

            best_acc = 0
            saliency_all = {}
            for iteration in range(command_line_args.num_iterations):

                mean_grad = []
                std_grad = []
                for idx, frame in enumerate(tqdm(video)):
                    conf.SR_dnn.net.train()
                    # save_img(frame, f'frames/sr{sec}_{idx}.png')
                    frame_orig = frame
                    if settings.backprop.tunable_config.cloudseg:
                        frame = conf.SR_dnn(frame.unsqueeze(0).to('cuda:1')).cpu()
                    # save_img(frame[0], f'frames/after_sr{sec}_{idx}.png')

                    reconstruction_loss = None
                    result = None
                    gt_frame = gt_video_new[idx].unsqueeze(0)



                    if command_line_args.loss_type == 'absolute_error':

                        result = app.inference(frame.detach(), detach=True, grad=False)
                        reconstruction_loss =  (gt_frame - frame).abs().mean()
                        reconstruction_loss.backward()




                    elif command_line_args.loss_type == 'saliency_error':
                        gt_frame = gt_video_new[idx]

                        # if settings.backprop.tunable_config.cloudseg:
                        #     frame = conf.SR_dnn(frame.unsqueeze(0).to('cuda:1')).cpu()

                        frame_detached = frame.detach()
                        frame_detached.requires_grad_()

                        result = app.inference(frame_detached, detach=False, grad=True, raw=True)
                        probs=F.softmax(result)
                        probs = probs[:, 1:, :, :]
                        # print(probs[0])

                        # filter out unrelated classes
                        sum_score = probs.sum() #+ 0.01 * torch.norm((frame_detached-gt_frame), p=2)
                        sum_score.backward()
                        # print(sum_score)
                        # print("Norm: ", torch.norm((frame_detached-gt_frame), p=2))
                        # print("Mean: ", frame_detached.grad.mean())
                        # print("Std: ", frame_detached.grad.std())

                        saliency = frame_detached.grad.cpu().detach()
                        frame_detached.grad = frame_detached.grad / frame_detached.grad.std()
                        # breakpoint()
                        (1*torch.norm((frame_detached-gt_frame), p=2)*saliency.std()).backward()
                        saliency = frame_detached.grad

                        image = T.ToPILImage()(frame.detach()[0].clamp(0, 1))
                        # print(saliency.shape)
                        # visualize_heat_by_summarywriter(image, saliency.abs().sum(dim=1, keepdim=True)[0][0], f'images/{sec}/{command_line_args.loss_type}_{sec}_{idx}_{iteration}', writer, 0, tile=False)
                        diff =  (frame-gt_frame)
                        reconstruction_loss = (saliency * diff).sum()

                        reconstruction_loss.backward()

                    elif command_line_args.loss_type == 'saliency_error_plus':
                        # frame.requires_grad_()
                        reconstruction_loss = (saliencies[idx] * (gt_frame - frame).abs()).mean()
                        reconstruction_loss = reconstruction_loss * command_line_args.alpha

                        if sec % 3 == 0:
                            reconstruction_loss = reconstruction_loss.to("cuda:0")
                            result = app.inference(frame, detach=False, grad=True, raw=True)
                            gt_result = app.inference(gt_frame, detach=False, grad=False, raw=False)
                            gt_result = app.transform_output(gt_result)
                            dice_loss = app.dice_loss(result, gt_result)
                            reconstruction_loss += dice_loss


                        reconstruction_loss.backward()

                    elif command_line_args.loss_type == 'feature_error':

                        # gt_result = app.inference(gt_frame, detach=False, grad=False, feature=True)
                        # frame.retain_grad()
                        #

                        # gt_result = gt_results[idx]
                        # print(frame.shape)
                        frame_detached = frame.detach()
                        frame_detached.requires_grad_()
                        result = app.inference(frame, detach=False, grad=True, raw=True)

                        gt_result = app.inference(gt_frame, detach=False, grad=False, raw=True)
                        feature_diffs = []
                        feature_diffs.append((gt_result - result).abs().mean())

                        reconstruction_loss = sum(feature_diffs)
                        reconstruction_loss.backward()

                        # image = T.ToPILImage()(frame.detach()[0].clamp(0, 1))
                        # image_FN = app.visualize(image, {'instances': inference_result['instances']})
                        # image_FN.save(f'{args.input.split("/")[-2]}_FN_filter/{command_line_args.loss_type}_{sec}_{idx}.jpg')
                        # visualize_heat_by_summarywriter(image_FN, saliency, f'{args.input.split("/")[-2]}_FN_filter/{command_line_args.loss_type}_{sec}_{idx}', writer, 0, tile=False)

                    elif command_line_args.loss_type == 'normal_update':
                        # frame.requires_grad_()

                        result = app.inference(frame, detach=False, grad=True, raw=True)
                        gt_result = app.inference(gt_frame, detach=False, grad=False, raw=False)
                        gt_result = app.transform_output(gt_result)
                        # print("Result shape: ", result.shape)
                        # print("GT Result shape: ", gt_result.shape)
                        reconstruction_loss = app.dice_loss(result, gt_result)+ command_line_args.alpha * torch.norm(frame-gt_frame, p=2)#torch.norm(frame-gt_frame, p=2) #- app.dice_loss(result, gt_result)
                        # reconstruction_loss = app.jaccard_loss(result, gt_result)
                        # reconstruction_loss = app.focal_loss(result, gt_result)
                        reconstruction_loss.backward()
                        # for i in range(gt_result.shape[1]):
                        #     print(torch.sum(gt_result[0, i, :, :] == 1))

                    elif command_line_args.loss_type == 'normal_update_saliency_plus':
                        diff = (frame-gt_frame).abs()
                        # if iteration == 0:
                        #     frame_detached = frame.detach()
                        #     frame_detached.requires_grad_()
                        #     result = app.inference(frame_detached, detach=False, grad=True, raw=True)
                        #
                        #     gt_result = app.inference(gt_frame, detach=False, grad=False, raw=False)
                        #     gt_result = app.transform_output(gt_result)
                        #     # print("Result shape: ", result.shape)
                        #     # print("GT Result shape: ", gt_result.shape)
                        #     dice_loss = app.dice_loss(result, gt_result) + command_line_args.alpha * torch.norm(frame.detach().cpu()-gt_frame, p=2)#torch.norm(frame-gt_frame, p=2) #- app.dice_loss(result, gt_result)
                        #     # for param in conf.SR_dnn.net.parameters():
                        #     #     param.requires_grad = False
                        #     dice_loss.backward(retain_graph=True)
                        #     #
                        #     # saliency =  frame_detached.grad
                        #     saliency =  frame_detached.grad
                        #     saliency_all[idx] = saliency
                        reconstruction_loss = (saliencies[idx] *  diff ).sum()
                        # print("Loss: ", reconstruction_loss.item())
                        reconstruction_loss.backward()
                    elif command_line_args.loss_type == 'normal_update_saliency':
                        frame_detached = frame.detach()
                        frame_detached.requires_grad_()
                        result = app.inference(frame_detached, detach=False, grad=True, raw=True)

                        gt_result = app.inference(gt_frame, detach=False, grad=False, raw=False)
                        gt_result = app.transform_output(gt_result)
                        # print("Result shape: ", result.shape)
                        # print("GT Result shape: ", gt_result.shape)
                        dice_loss = app.dice_loss(result, gt_result) + command_line_args.alpha * torch.norm(frame.detach().cpu()-gt_frame, p=2)#torch.norm(frame-gt_frame, p=2) #- app.dice_loss(result, gt_result)
                        # for param in conf.SR_dnn.net.parameters():
                        #     param.requires_grad = False
                        dice_loss.backward(retain_graph=True)
                        #
                        # saliency =  frame_detached.grad
                        saliency =  frame_detached.grad.abs()
                        diff = (frame-gt_frame).abs()
                        # if iteration == 0:
                        #     plot_pdf(torch.flatten(frame_detached.grad.abs().log()).detach().cpu().numpy(), args.second, idx, command_line_args.loss_type )
                        # reconstruction_loss = (saliency * torch.square(frame-gt_frame)).sum()
                        # reconstruction_loss = (saliency *  torch.square(frame-gt_frame)).abs().sum()
                        assert torch.equal(frame, frame - torch.zeros(frame.shape))

                        reconstruction_loss = (saliency *  diff ).sum()
                        # print("Loss: ", reconstruction_loss.item())
                        reconstruction_loss.backward()

                    elif command_line_args.loss_type == 'normal_update_saliency_block':
                        frame_detached = frame.detach()
                        frame_detached.requires_grad_()
                        result = app.inference(frame_detached, detach=False, grad=True, raw=True)

                        gt_result = app.inference(gt_frame, detach=False, grad=False, raw=False)
                        gt_result = app.transform_output(gt_result)
                        # print("Result shape: ", result.shape)
                        # print("GT Result shape: ", gt_result.shape)
                        dice_loss = app.dice_loss(result, gt_result) + command_line_args.alpha * torch.norm(frame.detach().cpu()-gt_frame, p=2)#torch.norm(frame-gt_frame, p=2) #- app.dice_loss(result, gt_result)
                        # for param in conf.SR_dnn.net.parameters():
                        #     param.requires_grad = False
                        dice_loss.backward(retain_graph=True)
                        #
                        # saliency =  frame_detached.grad
                        saliency = frame_detached.grad.abs().sum(dim=1, keepdim=True)

                        diff = (frame-gt_frame).abs().sum(dim=1, keepdim=True)

                        compressed_saliency = compress_saliency(saliency)

                        # print(torch.unique(compressed_saliency))
                        image = T.ToPILImage()(frame.detach()[0].clamp(0, 1))
                        # visualize_heat_by_summarywriter(image, compressed_saliency[0][0], f'images/{sec}/{command_line_args.loss_type}_{sec}_{idx}_{iteration}', writer, 0, tile=False)
                        reconstruction_loss = (compressed_saliency *  diff ).sum()
                        # print("Loss: ", reconstruction_loss.item())
                        reconstruction_loss.backward()
                    elif command_line_args.loss_type == 'mse':
                        frame_detached = frame.detach()
                        frame_detached.requires_grad_()

                        # dice_loss = app.dice_loss(result, gt_result) + command_line_args.alpha * torch.norm(frame - gt_frame, p=2)
                        mse_loss = torch.norm(frame_detached-gt_frame, p=2)

                        mse_loss.backward()
                        saliency = frame_detached.grad.abs().sum(dim=1, keepdim=True)
                        kernel = torch.ones([1, 1, command_line_args.average_window_size, command_line_args.average_window_size])
                        saliency = F.conv2d(saliency, kernel, stride=1, padding=(command_line_args.average_window_size - 1) // 2)
                        reconstruction_loss = (saliency * (gt_frame - frame).abs()).mean()
                        reconstruction_loss.backward()
                        image = T.ToPILImage()(frame.detach()[0].clamp(0, 1))
                        # visualize_heat_by_summarywriter(image, saliency[0][0], f'images/{sec}/{command_line_args.loss_type}_{sec}_{idx}', writer, 0, tile=False)

                    elif command_line_args.loss_type == 'mse_one_norm':
                        frame_detached = frame.detach()
                        frame_detached.requires_grad_()

                        # dice_loss = app.dice_loss(result, gt_result) + command_line_args.alpha * torch.norm(frame - gt_frame, p=2)
                        mse_loss = torch.norm(frame_detached-gt_frame, p=1)

                        mse_loss.backward()
                        saliency = frame_detached.grad.abs().sum(dim=1, keepdim=True)
                        print(torch.unique(frame_detached.grad))
                        # print(torch.flatten(frame_detached.grad.abs()).detach().cpu().numpy())
                        # if iteration == 0:
                        #     plot_cdf(torch.flatten(frame_detached.grad.abs()).detach().cpu().numpy(), args.second, idx, command_line_args.loss_type)
                        # kernel = torch.ones([1, 1, command_line_args.average_window_size, command_line_args.average_window_size])
                        # saliency = F.conv2d(saliency, kernel, stride=1, padding=(command_line_args.average_window_size - 1) // 2)
                        frame.requires_grad_()
                        reconstruction_loss = (saliency * (gt_frame - frame).abs()).sum()
                        reconstruction_loss.backward()
                        # image = T.ToPILImage()(frame.detach()[0].clamp(0, 1))

                    elif command_line_args.loss_type == 'normal_update_saliency_test':
                        frame_detached = frame.detach()
                        frame_detached.requires_grad_()
                        result = app.inference(frame_detached, detach=False, grad=True, raw=True)

                        gt_result = app.inference(gt_frame, detach=False, grad=False, raw=False)
                        gt_result = app.transform_output(gt_result)
                        # print("Result shape: ", result.shape)
                        # print("GT Result shape: ", gt_result.shape)
                        dice_loss = app.dice_loss(result, gt_result) + command_line_args.alpha * torch.norm(frame-gt_frame, p=2)#torch.norm(frame-gt_frame, p=2) #- app.dice_loss(result, gt_result)
                        # for param in conf.SR_dnn.net.parameters():
                        #     param.requires_grad = False
                        dice_loss.backward(retain_graph=True)
                        #
                        saliency = frame_detached.grad
                        reconstruction_loss = (saliency * (frame).abs()).sum()

                        reconstruction_loss.backward()
                        # # print("Gradient mean: ", frame_detached.grad.abs().mean())
                        # # mean_grad.append( frame_detached.grad.abs().mean().item())
                        # # std_grad.append(torch.std(frame_detached.grad.abs()).item())
                        # saliency = frame_detached.grad.abs().sum(dim=1, keepdim=True)
                        # kernel = torch.ones([1, 1, command_line_args.average_window_size, command_line_args.average_window_size])
                        # saliency = F.conv2d(saliency, kernel, stride=1, padding=(command_line_args.average_window_size - 1) // 2)
                        # image = T.ToPILImage()(frame.detach()[0].clamp(0, 1))
                        # # if os.path.exists(f'images/{sec}') is False:
                        # #     os.makedirs(f'images/{sec}')
                        # if iteration % 5 == 0:
                        #     visualize_heat_by_summarywriter(image, saliency[0][0].abs(), f'images/{sec}/{command_line_args.loss_type}_{sec}_{idx}_{iteration}', writer, 0, tile=False)



                    else:
                        raise NotImplementedError






                    if settings.backprop.early_optimize:
                        # logger.info('Early optimize')
                        # for name, param in conf.SR_dnn.net.named_parameters():
                        #     if param.grad is not None and name == 'b3.b1.body.0.weight':
                        #         print(name)
                        #         print(torch.sum(param.grad))
                        optimizer.step()
                        optimizer.zero_grad()


                print(mean_grad)
                print(std_grad)

                for key in settings.backprop.tunable_config.keys():
                    if key == 'cloudseg':
                        # already optimized when backwarding.
                        continue
                    optimize(args, key, video.grad)
                if iteration % 5 == 0:
                    performance = examine(args, gt_args, app, db)
                    logger.info('mAP: %.3f', performance['f1'])
                    if performance['f1'] > best_acc:
                        best_acc = performance['f1']

                        # torch.save(conf.SR_dnn.net.state_dict(), f'checkpoints_test/carn_m_{args.second}_{command_line_args.loss_type}.pth')
                    for idx, frame in enumerate(tqdm(video)):
                        conf.SR_dnn.net.eval()
                        gt_frame = gt_video_new[idx].unsqueeze(0)
                        if settings.backprop.tunable_config.cloudseg:
                            with torch.no_grad():

                                # new_state_dict = torch.load(f'checkpoints/carn_m_{args.second}_{command_line_args.loss_type}.pth')
                                # conf.SR_dnn.net.load_state_dict(new_state_dict)
                                frame = conf.SR_dnn(frame.unsqueeze(0).to('cuda:1')).cpu()
                        image = T.ToPILImage()(frame.cpu().detach()[0])
                        # result = app.inference(frame, detach=True, grad=False)
                        # image = app.visualize(image, {'instances': result}, command_line_args)
                        print(f'{args.input.split("/")[-2]}_FN_filter/inference_{sec}_{idx}.jpg')
                        image.save(f'images_Saliency/inference_{sec}_{idx}_{command_line_args.loss_type}.jpg')
            if not settings.backprop.early_optimize:

                optimizer.step()
                optimizer.zero_grad()






        # objective = (settings.backprop.sum_score_mean_weight * average_sum_score + settings.backprop.std_score_mean_weight * average_std_score_mean)
        # true_obj = (settings.backprop.sum_score_mean_weight * true_average_score + settings.backprop.std_score_mean_weight * true_average_std_score_mean  - settings.backprop.compute_weight * interpolated_fr.detach().item())


        # for idx, frame in enumerate(tqdm(video)):
        #     conf.SR_dnn.net.eval()
        #     gt_frame = gt_video_new[idx].unsqueeze(0)
        #     if settings.backprop.tunable_config.cloudseg:
        #         with torch.no_grad():
        #
        #             new_state_dict = torch.load(f'checkpoints/carn_m_{args.second}_{command_line_args.loss_type}.pth')
        #             conf.SR_dnn.net.load_state_dict(new_state_dict)
        #             frame = conf.SR_dnn(frame.unsqueeze(0).to('cuda:1')).cpu()
        #     image = T.ToPILImage()(frame.cpu().detach()[0])
        #     # result = app.inference(frame, detach=True, grad=False)
        #     # image = app.visualize(image, {'instances': result}, command_line_args)
        #     print(f'{args.input.split("/")[-2]}_FN_filter/inference_{sec}_{idx}.jpg')
        #     image.save(f'images/inference_{sec}_{idx}.jpg')
            # image.save(f'{args.input.split("/")[-2]}_FN_filter/inference_{sec}_{idx}.jpg')
            # image = T.ToPILImage()(gt_frame[0])
            # gt_result = app.inference(gt_frame, detach=True, grad=False)
            # image = app.visualize(image, {'instances': gt_result}, command_line_args)
            # image.save(f'{args.input.split("/")[-2]}_FN_filter/gt_{sec}_{idx}.jpg')
        performance = examine(args, gt_args, app, db)

        iou_scores.append(performance['f1'])
        logger.info('mAP: %.3f', performance['f1'])

        # logger.info('fn hidden ratio: %.3f', performance['fn_hidden_ratio'])

        # logger.info('F1 relaxed: %.3f', performance['f1_debug'])
        # logger.info('recall relaxed: %.3f', performance['recall_debug'])
        # print(confidence_score_dict)
        # db['cs_distribution'].insert_one(confidence_score_dict)
        # logger.info('True : %.3f, Tru: %.3f, Tru: %.3f, Tru: %.3f', true_average_score, true_average_std_score_mean, true_average_bw, true_obj)

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



        logger.info(f'Current config: {conf.state2config(state)}')

        logger.info(f'Current state: {state}')

        # choose = conf.random_serialize(video_name, conf.state2config(state))


        # # logger.info('Choosing %s', choose)

        # with open(args.output, 'a') as f:
        #     f.write(yaml.dump([{
        #         'sec': sec,
        #         # 'choice': choose,
        #         'config': conf.state2config(state, serialize=True),
        #         'true_average_bw': true_average_bw,
        #         'true_average_score': true_average_score,
        #         'true_average_f1': true_average_f1,
        #         'fuse_obj': fuse_obj.item(),
        #         'true_obj': true_obj,
        #         'average_sum_score': average_sum_score.item(),
        #         'average_std_score_mean': average_std_score_mean.item(),
        #         'average_range_score_mean': ((sum_score.max() - sum_score.min()) / interpolated_fr).item(),
        #         'average_abs_score_mean': (sum_score - sum_score.mean()).abs().mean().item(),
        #         'state': state_str,
        #         # 'all_states': list(conf.serialize_all_states(args.input, conf.state2config(state, serialize=True), 1., conf.serialize_order)),
        #         # 'qp_grad': state['qp'].grad.item()
        #     }]))








        # set_trace()

    # for idx, (hqs, lqs) in enumerate(zip(read_video(args.hq, args), read_video(args.lq, args))):

    #     hqs = torch.cat([i[1] for i in hqs])
    #     lqs = torch.cat([i[1] for i in lqs])

    #     # frames = fr(q(hqs, lqs))
    #     frames = q(hqs, lqs)
    #     # frames = fr(hqs)


    #     for frame, hq in zip(frames, hqs):

    #         progress_bar.update()

    #         with torch.no_grad():
    #             result = app.inference(frame.unsqueeze(0), detach=True)
    #         # with torch.no_grad():
    #         #     hq_result = app.inference(hq.unsqueeze(0), detach=True)
    #         inference_results[fid] = result

    #         if idx % args.freq == 0:
    #             activation = app.activation(frame.unsqueeze(0))
    #             activation.backward(retain_graph=True)

    #         fid += 1

    #     if idx % args.freq == 0:
    #         # fr.step()
    #         q.step()

    #         image = F.interpolate(hqs, size=(480, 640))
    #         image = T.ToPILImage()(image[0])
    #         image = app.visualize(image, result, args)
    #         writer.add_image('inference', T.ToTensor()(image), fid)

    #         q.visualize(hqs[0], fid)

    #         means.append(q.q.detach().mean())
    iou_avg = sum(iou_scores) / len(iou_scores)

    print(np.sort(all_fn_scores))
    print(command_line_args.approach)
    x = {
    'input': command_line_args.input,
    'loss_type': command_line_args.loss_type,
    'command_line_args.approach': command_line_args.approach,
    }
    x.update({'all_scores': pickle.dumps(all_scores)})
    x.update({'all_fn_scores': pickle.dumps(all_fn_scores)})
    x.update({'all_fp_scores': pickle.dumps(all_fp_scores)})

    db['conf'].insert_one(x)
    print("Mean score: ", iou_avg)
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
    parser.add_argument(
        '--tile_size',
        type=int,
        default=7
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5
    )
    # parser.add_argument(
    #     '--gamma',
    #     type=float,
    #     help='Adjust the luminance.',
    #     default=1.5,
    # )

    args = parser.parse_args()

    main(args)
