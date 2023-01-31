"""
    Compress the video through gradient-based optimization.
"""

import sys
sys.path.append('/datamirror/kuntai/code/diff/')

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
import itertools

from copy import deepcopy


sns.set()
set_seed()
logger = logging.getLogger("chameleon")

def generate_config_space(args):

    lists = []

    for key in settings.configuration_space:
        lists.append([(key, idx, val) for idx, val in enumerate(settings.configuration_space[key])])

    configs = itertools.product(*lists)
    config_args = []
    for config in configs:
        new_args = deepcopy(args)
        for key, idx, val in config:
            new_args[key] = val
            new_args[key + 'idx'] = idx
        config_args.append(new_args)

    return config_args


def get_inference_stats(results, conf_thresh):

    get_sum = lambda x: x['instances'].scores.sum()
    get_sigmoid = lambda x: ((x['instances'].scores - conf_thresh) * 20).sigmoid().sum()

    return {
        'inference_sum': torch.tensor(list(map(get_sum, results.values()))).mean().item(),
        'inference_sigmoid': torch.tensor(list(map(get_sigmoid, results.values()))).mean().item(),
    }




def profile(command_line_args, gt_args, app, db, conf_thresh, stats_file):

    examine(gt_args, gt_args, app, db, key='profile')

    for config_arg in generate_config_space(gt_args):
        stats = examine(config_arg, gt_args,app,db,key='profile')
        inference_stats = get_inference_stats(pickle.loads(stats['my_inference_result']), conf_thresh)


        ret = inference_stats
        ret['f1'] = stats['f1']
        for key in ['second', 'input', 'qpidx', 'residx', 'bframebiasidx']:
            ret[key] = stats['my_args'][key]

        with open(stats_file, 'a') as f:
            f.write(yaml.dump([ret]))

        
        





def main(command_line_args):
    


    # a bunch of initialization
    torch.set_default_tensor_type(torch.FloatTensor)

    db = pymongo.MongoClient("mongodb://localhost:27017/")[settings.collection_name]

    app = DNN_Factory().get_model(settings.backprop.app)
    conf_thresh = settings[app.name].confidence_threshold

    # writer = SummaryWriter(f"runs/{command_line_args.input}/{command_line_args.approach}")


    logger.info("Application: %s", app.name)
    logger.info("Input: %s", command_line_args.input)

    optimal_args = munchify(settings.ground_truths_config.to_dict()) 
    optimal_args.input = command_line_args.input
    optimal_args.command_line_args = vars(command_line_args)
    optimal_args.settings = settings.as_dict()
    
    current_args = None

    gt_args = deepcopy(optimal_args)
    # constraint optimal args's QP into the configuration space.
    # optimal_args.qp = 20

    for second in tqdm(range(command_line_args.start, command_line_args.end)):

        gt_args.second = second
        profile(command_line_args, gt_args, app, db, conf_thresh, command_line_args.stats)

        



        


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
        "--app", 
        type=str, 
        help="The name of the model.", 
        default='EfficientDet-d2',
    )

    parser.add_argument(
        '--stats', 
        type=str,
        required=True,
        help='Dump the inference stats to this stats file'
    )

    # parser.add_argument(
    #     '--gamma',
    #     type=float,
    #     help='Adjust the luminance.',
    #     default=1.5,
    # )

    args = parser.parse_args()

    main(args)
