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
# from inference import inference, encode
# from examine import examine
from utils.inference import inference, examine
from utils.encode import encode
import pymongo
from munch import *
from utils.seed import set_seed

from copy import deepcopy


sns.set()
set_seed()
logger = logging.getLogger("chameleon")



def profile(command_line_args, previous_arg, gt_args, app, db, compute):

    stat = examine(previous_arg, gt_args, app, db, key='profile')
    gt_bw = stat['my_video_config']['bw']
    gt_compute = len(stat['my_video_config']['encoded_frames'])
    objective = (1-stat['f1']) \
        + settings.backprop.bw_weight * stat['my_video_config']['bw'] / gt_bw
        # + settings.backprop.compute_weight * len(stat['encoded_frames']) / gt_compute

    logger.info(f'Initial objective: {objective:.4f}')
    
    optimal_args = deepcopy(previous_arg)
    optimal_stat = stat


    for key in settings.configuration_space:
        
        
        if command_line_args.enable_top3 and key not in ['res', 'qp', 'bframebias']:
            continue

        args = deepcopy(optimal_args)

        for idx, val in enumerate(settings.configuration_space[key]):
            
            if idx == len(settings.configuration_space[key]) - 1:
                continue # The last configuration is just a sentinal.
            if idx % command_line_args.downsample_factor != 0:
                continue # downsample the configuration space uniformly

            logger.info(f'Searching {key}:{val}')

            args[key] = val

            stat = examine(args, gt_args, app, db, key='profile')
            new_objective = (1-stat['f1']) \
                + settings.backprop.bw_weight * stat['my_video_config']['bw'] / gt_bw 
                # + settings.backprop.compute_weight * len(stat['encoded_frames']) / gt_compute

            logger.info(f'Objective: %.4f 1-F1: %.4f BW: %.4f Com: %.4f' %(
                new_objective,
                1- stat['f1'],
                stat['my_video_config']['bw'] / gt_bw,
                len(stat['my_video_config']['encoded_frames']) / gt_compute
            ))
            
            compute['compute'] = compute['compute'] + len(stat['my_video_config']['encoded_frames'])

            if new_objective < objective:
                logger.info('Update objective.')
                objective = new_objective
                optimal_args = deepcopy(args)
                optimal_stat = stat


    for key in settings.configuration_space:
        logger.info(f'{key}: {optimal_args[key]}')

    for stat_key in ['f1', 'bw', 'encoded_frames']:
        if stat_key == 'f1':
            logger.info(f'{stat_key}: {optimal_stat[stat_key]}')
        else:
            logger.info(f'{stat_key}: {optimal_stat["my_video_config"][stat_key]}')
        
    delay = (compute['compute'] + settings.compute_limit - 1) // settings.compute_limit


    # logger.info(f'Pick {optimal_args}')
    # logger.info(f'Stats: {stat}')
    # breakpoint()

    return optimal_args, delay









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
    
    current_args = None

    gt_args = deepcopy(optimal_args)
    
    # constraint optimal args's QP into the configuration space.
    # optimal_args.qp = 20
    
    next_profile_second = 0

    for second in range(command_line_args.start, command_line_args.end):

        optimal_args.second = second
        gt_args.second = second
        if current_args is not None:
            current_args.second = second
        
        compute = {'compute': 0}
        profile_flag = False

        if second == next_profile_second:


            profile_flag = True
            '''
                Previous profiling finished.
                Refresh the config and launch the new profiling.
            '''
            current_args = optimal_args.copy()
            current_args.second = second
            optimal_args, delay = profile(command_line_args, optimal_args, gt_args, app, db, compute)
            if settings.chameleon.immediate_profile:
                current_args = optimal_args.copy()
                current_args.second = second
            
            next_profile_second = second + delay
            
            logger.info('Next profiling second is %d', next_profile_second)

        stat = examine(current_args, gt_args, app, db, profile=profile_flag)
        norm_bw = 1.
        f1 = 1.0
        
        # if second % args.frequency != 0:
        #     compute['compute'] = len(stat['encoded_frames'])
        #     norm_bw = stat['norm_bw']
        #     f1 = stat['f1']
            
        # db['cost'].insert_one({
        #     'command_line_args': vars(command_line_args), 
        #     'settings': settings.to_dict(),
        #     'compute': compute['compute'] / settings.segment_length,
        #     'norm_bw': norm_bw,
        #     'second': second,
        #     'f1': f1,
        # })

        



        


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
        '--downsample_factor',
        help='The downsample factor of the configuration space',
        required=True,
        type=int,
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
    
    parser.add_argument(
        '--enable_top3',
        default=False,
        action='store_true',
    )
    # parser.add_argument(
    #     '--gamma',
    #     type=float,
    #     help='Adjust the luminance.',
    #     default=1.5,
    # )

    args = parser.parse_args()

    main(args)
