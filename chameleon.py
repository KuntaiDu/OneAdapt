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

from copy import deepcopy


sns.set()
set_seed()
logger = logging.getLogger("chameleon")



def profile(command_line_args, previous_arg, gt_args, app, db):


    stat = examine(previous_arg, gt_args, app, db)
    gt_bw = stat['bw']
    gt_compute = len(stat['encoded_frames'])
    objective = (1-stat['f1']) \
        + settings.backprop.bw_weight * stat['bw'] / gt_bw \
        + settings.backprop.compute_weight * len(stat['encoded_frames']) / gt_compute

    logger.info(f'Initial objective: {objective:.4f}')
    
    optimal_args = deepcopy(previous_arg)


    for key in settings.configuration_space:

        args = deepcopy(optimal_args)

        for val in settings.configuration_space[key]:

            logger.info(f'Searching {key}:{val}')

            args[key] = val

            stat = examine(args, gt_args, app, db, key='profile')
            new_objective = (1-stat['f1']) \
                + settings.backprop.bw_weight * stat['bw'] / gt_bw \
                + settings.backprop.compute_weight * len(stat['encoded_frames']) / gt_compute

            logger.info(f'Objective: {new_objective:.4f}')

            if new_objective < objective:
                logger.info('Update objective.')
                objective = new_objective
                optimal_args = args

    return optimal_args









def main(command_line_args):


    # a bunch of initialization.
    
    torch.set_default_tensor_type(torch.FloatTensor)

    db = pymongo.MongoClient("mongodb://localhost:27017/")[settings.collection_name]

    app = DNN_Factory().get_model(settings.backprop.app)

    writer = SummaryWriter(f"runs/{command_line_args.input}/{command_line_args.approach}")


    logger.info("Application: %s", app.name)
    logger.info("Input: %s", command_line_args.input)
    logger.info("Approach: %s", command_line_args.approach)
    progress_bar = enlighten.get_manager().counter(
        total=command_line_args.end - command_line_args.start,
        desc=f"{command_line_args.input}",
        unit="10frames",
    )

    optimal_args = munchify(settings.ground_truths_config.to_dict()) 
    optimal_args.input = command_line_args.input
    optimal_args.command_line_args = vars(command_line_args)
    optimal_args.settings = settings.as_dict()

    gt_args = deepcopy(optimal_args)

    for second in range(command_line_args.start, command_line_args.end):

        optimal_args.second = second
        gt_args.second = second

        if second % args.frequency == 0:

            optimal_args = profile(command_line_args, optimal_args, gt_args, app, db)

        examine(optimal_args, gt_args, app, db)

        



        


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
        '-freq',
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
