"""
    Compress the video through gradient-based optimization.
"""

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
from munch import *
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
                            grad_reducto_expensive)
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


conf.space = munchify(settings.configuration_space.to_dict())
state = munchify({})

logger = logging.getLogger("Server")

conf.serialize_order = list(settings.backprop.tunable_config.keys())

# sanity check
for key in settings.backprop.frozen_config.keys():
    assert (
        key not in conf.serialize_order
    ), f"{key} cannot both present in tunable config and frozen config."


def read_expensive_from_config(
    gt_args, state, app, db, command_line_args, train_flag=False
):

    average_video = None
    average_bw = 0
    sum_prob = 0

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
            if "reducto" in key:
                if train_flag and settings.backprop.reducto_expensive_optimize:
                    # remove frame filtering for training purpose.
                    del args[key]
                else:
                    # this is reducto encoding.
                    args.tag = "reducto"
        if "tag" not in list(args.keys()):
            # this is normal encoding.
            args.tag = "mpeg"

        if "macroblocks" in state:
            args.macroblocks = state["macroblocks"]

        # encode
        args.command_line_args = vars(command_line_args)
        args.settings = settings.to_dict()
        video_name, video_config = encode(args)
        logger.info("Ours bandwidth: %d", video_config["bw"])
        video = read_video_to_tensor(video_name)

        # update statistics of random choice.
        inference(args, db, app, video_name, video_config)
        stat = examine(args, gt_args, app, db)

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
    if settings.backprop.visualize:
        visualize_folder = Path('debug/' + command_line_args.input).parent
        visualize_folder.mkdir(exist_ok=True, parents=True)
        results_vis = VideoVisualizer(f"{visualize_folder}/{command_line_args.approach}_results.mp4")
        errors_vis = VideoVisualizer(f"{visualize_folder}/{command_line_args.approach}_errors.mp4")

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



    # the saliency of last time
    last_saliency = None
    '''
        Main loop
    '''
    for sec in tqdm(
        range(command_line_args.start, command_line_args.end),
        unit="sec",
        desc="progress",
    ):

        # the text for visualization
        vis.text = ""
        logger.info("Input: %s", command_line_args.input)

        # get the ground truth configuration and the camera-side ground truth video
        gt_args = munchify(settings.ground_truths_config.to_dict())
        gt_args.update(
            {"input": command_line_args.input, "second": sec,}
        )
        gt_video_name, gt_video_config = encode(gt_args)
        gt_video = read_video_to_tensor(gt_video_name)
        logger.info("Ground truth bandwidth: %d", gt_video_config["bw"])
        # update the mean and the std of reducto thresholds.
        for key in state.keys():
            if "reducto" in key:
                with Timer("reducto_update_mean_std_timer", logger):
                    reducto_update_mean_std(gt_video, state, gt_args, db)
                break
            
            
        state['reducto_area_bias'][()] = command_line_args.area
        state['reducto_pixel_bias'][()] = command_line_args.area
        # encode and inference on the current video.
        train_flag=False
        stat, args, server_video = read_expensive_from_config(
            gt_args, state, app, db, command_line_args, train_flag
        )
        yaml_result = {
            'bandwidth': stat['my_video_config']['bw'],
            'compute': len(stat['my_video_config']['encoded_frames']),
            'input': args.input,
            'second': args.second,
            'f1': stat['f1'],
        }
        with open(f'stats/{command_line_args.approach}.yaml', 'a') as f:
            f.write(yaml.dump([yaml_result]))


if __name__ == "__main__":

    # set the format of the logger
    formatter = coloredlogs.ColoredFormatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s",
        datefmt="%H:%M:%S",
    )

    handler = TqdmLoggingHandler()
    handler.setFormatter(formatter)
    logging.basicConfig(handlers=[handler], level="INFO")

    parser = argparse.ArgumentParser()

    parser.add_argument("--loss_type", type=str, required=True)

    parser.add_argument(
        "-i", "--input", help="The format of input video.", type=str, required=True
    )

    parser.add_argument(
        "--start", help="The total secs of the video.", required=True, type=int
    )

    parser.add_argument(
        "--end", help="The total secs of the video.", required=True, type=int
    )

    parser.add_argument(
        "--num_iterations", help="The total secs of the video.", required=True, type=int
    )
    
    parser.add_argument(
        "--area", type=float, required=True,
    )

    parser.add_argument(
        "--frequency", help="The total secs of the video.", required=True, type=int
    )

    parser.add_argument(
        "--app", type=str, help="The name of the model.", default="EfficientDet-d2",
    )

    parser.add_argument(
        "--average_window_size",
        type=int,
        help="The window size for saliency averaging",
        default=17,
    )

    parser.add_argument("--approach", type=str, required=True)

    args = parser.parse_args()

    main(args)
