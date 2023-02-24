import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import gc
import logging
import pickle
import random
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from pdb import set_trace
from typing import Tuple

import coloredlogs
import enlighten
import matplotlib.pyplot as plt
import pymongo
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import yaml
from detectron2.structures.instances import Instances
from munch import munchify
from PIL import Image, ImageDraw, ImageFont
from torch.utils.tensorboard import SummaryWriter
from torchvision import io
from tqdm import tqdm

import utils.config_utils as conf
import utils.video_visualizer as vis
from config import settings
from dnn.dnn import DNN
from dnn.dnn_factory import DNN_Factory
from utils.encode import encode, tile_mask
from utils.gradient import (grad, grad_macroblocks, grad_reducto_cheap,
                            grad_reducto_expensive, postprocess_saliency)
from utils.inference import examine, inference
from utils.reducto import (reducto_feature2meanstd, reducto_process,
                           reducto_update_mean_std)
from utils.seed import set_seed
from utils.timer import Timer
from utils.tqdm_handler import TqdmLoggingHandler
# from utils.results import write_results
from utils.video_reader import (read_video, read_video_config,
                                read_video_to_tensor)
from utils.video_visualizer import VideoVisualizer
from utils.visualize_utils import visualize_heat_by_summarywriter

sns.set()
set_seed()



optimizer = None
state = munchify({})

logger = logging.getLogger("Server")













def get_configuration(
    gt_args, app, db, command_line_args
):

    global state


    for args in conf.serialize_most_expensive_state(
        gt_args.copy(), conf.state2config(state), conf.serialize_order
    ):

        args = args.copy()
        # use ground truth here. For debugging purpose.
        if hasattr(args, 'tag'):
            del args.tag
        for key in list(args.keys()):
            if "qp" in key and "macroblocks" in state:
                del args.qp
            # if "reducto" in key:
            #     if train_flag and settings.backprop.reducto_expensive_optimize:
            #         # remove frame filtering for training purpose.
            #         del args[key]
            #     else:
            #         # this is reducto encoding.
            #         args.tag = "reducto"
        if "tag" not in list(args.keys()):
            # this is normal encoding.
            args.tag = "mpeg"

        if "macroblocks" in state:
            args.macroblocks = state["macroblocks"]

        # encode
        args.command_line_args = vars(command_line_args)
        args.settings = settings.to_dict()
        return args




def init():

    global state, optimizer
    
    # initialize optimizer
    placeholder = torch.tensor([1.0])
    placeholder.requires_grad = True
    placeholder.grad = torch.zeros_like(placeholder)
    # avoid no parameter to optimize error
    parameters = [{"params": placeholder, "lr": 0.0}]
    for key in conf.serialize_order:
        if key == "cloudseg":
            for param in conf.SR_dnn.net.parameters():
                param.requires_grad = True
            parameters.append(
                {
                    "params": conf.SR_dnn.net.parameters(),
                    "lr": settings.backprop.tunable_config_lr[key],
                }
            )
            continue
        elif key == "macroblocks":
            # will directly solve the closed-form solution. No lr needed.
            state[key] = (
                torch.ones(settings.backprop.macroblock_shape).int()
                * settings.backprop.qps[0]
            )
            continue
        elif "reducto" in key:
            # will directly solve the closed-form solution. No lr needed.
            state[key] = torch.tensor(settings.backprop.tunable_config[key])
            continue

        lr = settings.backprop.lr
        if key in settings.backprop.tunable_config_lr.keys():
            lr = settings.backprop.tunable_config_lr[key]

        state[key] = torch.tensor(settings.backprop.tunable_config[key])
        state[key].requires_grad = True
        parameters.append({"params": state[key], "lr": lr})

    # build optimizer
    optimizer = torch.optim.Adam(parameters, betas=(0.5, 0.5))




def update_configuration(sec, command_line_args, video, gt_video, gt_args, db, app, config, gt_video_config, gt_results):

    global state, optimizer

    server_video = video
    conf_thresh = settings[app.name].confidence_threshold
    args = config

    for key in state.keys():
        if "reducto" in key:
            with Timer("reducto_update_mean_std_timer", logger):
                reducto_update_mean_std(gt_video, state, gt_args, db)
            break

    train_flag = (
        settings.backprop.train
        and (sec - command_line_args.start) % command_line_args.frequency == 0
    )
    if train_flag:

        saliencies = {}
        raw_saliencies_tensor = []
        saliencies_tensor = []
        inference_results = {}
        camera_video = server_video
        compute = None
        reducto_encode_fids = None

        if (
            any("reducto" in key for key in state.keys())
            and not settings.backprop.reducto_expensive_optimize
        ):
            # get the differentiable reducto processed video on the camera
            camera_video, reducto_encode_fids = reducto_process(gt_video, state, gt_args, db)

        # calculate saliency on each frame.
        for idx, frame in enumerate(
            tqdm(server_video, unit="frame", desc="saliency")
        ):

            # only backprop on the last frame.
            if idx != len(server_video) - 1:
                continue

            if reducto_encode_fids is not None and idx not in reducto_encode_fids:
                inference_results[idx] = inference_results[idx-1]
                raw_saliencies_tensor.append(raw_saliencies_tensor[idx-1])
                saliencies[idx] = saliencies[idx-1]
                saliencies_tensor.append(saliencies_tensor[idx-1])
                continue
            gt_frame = gt_video[idx]
            frame = frame.unsqueeze(0)

            frame_detached = frame.detach()
            frame_detached.requires_grad_()
            result = app.inference(frame_detached, detach=False, grad=True)
            # filter out unrelated classes
            result = app.filter_result(result, confidence_check=False)
            score = result["instances"].scores
            if settings.backprop.saliency_type == "sigmoid":
                sum_score = ((score - conf_thresh) * 20).sigmoid().sum()
            elif settings.backprop.saliency_type == "sum":
                sum_score = score[score > conf_thresh].sum()
            # score_inds = (conf_lb < score) & (score < conf_ub)
            # logger.info('%d objects need for optimization.' % score_inds.sum())
            # score = score[score_inds]
            # sum_score = torch.sum(score)
            inference_results[idx] = result
            sum_score.backward()

            saliency = postprocess_saliency(frame_detached.grad)


            raw_saliencies_tensor.append(saliency.clone())

        # video.requires_grad_()
        for iteration in range(command_line_args.num_iterations):

            reducto_optimized = False
            for key in settings.backprop.tunable_config.keys():
                if key == "cloudseg":
                    # already optimized when backwarding.
                    continue
                elif key == "macroblocks":
                    grad_macroblocks(
                        state,
                        args,
                        key,
                        torch.cat(raw_saliencies_tensor),
                        gt_video_config,
                        gt_video,
                    )
                    continue
                elif "reducto" in key:
                    if not reducto_optimized:
                        if settings.backprop.reducto_expensive_optimize:
                            grad_reducto_expensive(
                                state, gt_video, gt_results, app, gt_args, db
                            )
                        else:
                            grad_reducto_cheap(
                                state, gt_video, torch.cat(raw_saliencies_tensor), gt_args, db
                            )

                        reducto_optimized = True

                    continue
                grad(
                    state,
                    args,
                    key,
                    torch.cat(raw_saliencies_tensor),
                    gt_video_config,
                    gt_video,
                )

            optimizer.step()
            optimizer.zero_grad()

            # round to [0,1]
            for key in conf.serialize_order:
                if key == "cloudseg":
                    continue
                if key == "macroblocks":
                    continue
                if "reducto" in key:
                    continue
                state[key].requires_grad = False
                if state[key] > 1.0:
                    state[key][()] = 1.0
                if state[key] < 1e-7:
                    state[key][()] = 1e-7
                state[key].requires_grad = True

            # print out current state
            
            state_str = ""
            for key in conf.serialize_order:
                if key == "cloudseg":
                    param = list(conf.SR_dnn.net.parameters())[0]
                    logger.info(
                        "CloudSeg: gradient of layer 0 mean: %.3f, std: %.3f",
                        param.grad.mean(),
                        param.grad.std(),
                    )
                    continue
                if key == "macroblocks":
                    param = state[key]
                    logger.info("Macroblocks: mean: %.3f", param.float().mean())
                    continue
                if "reducto" in key:
                    param = state[key]
                    logger.info(f"{key}: %.3f", param.float().mean())
                    continue
                state_str += "%s : %.3f, grad: %.7f\n" % (
                    key,
                    state[key],
                    state[key].grad,
                )

            logger.debug(f"Current state: {state}")