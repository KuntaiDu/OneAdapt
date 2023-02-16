"""
    Compress the video through gradient-based optimization.
"""



import sys
from dynaconf import Dynaconf
import os
settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=[os.environ['SETTINGS_FILE']],
)
sys.path.append(settings.root_dir)
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
from utils.encode import encode, tile_mask, generate_mask_from_regions
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
from detectron2.structures.boxes import pairwise_iou
from utils.bbox_utils import center_size

from utils.accmpegmodel import FCN

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




def main(command_line_args):

    # a bunch of initialization.

    torch.set_default_tensor_type(torch.FloatTensor)

    db = pymongo.MongoClient("mongodb://localhost:27017/")[settings.collection_name]

    app = DNN_Factory().get_model(settings.dds.app)

    mask_generator = FCN()
    mask_generator.load(settings.accmpeg.model_path)
    mask_generator.cuda()

    conf_thresh = settings[app.name].confidence_threshold

    logger.info("Application: %s", app.name)
    logger.info("Input: %s", command_line_args.input)
    logger.info("Approach: %s", command_line_args.approach)

    # visualizer
    visualize_folder = Path('debug/' + command_line_args.input).parent
    visualize_folder.mkdir(exist_ok=True, parents=True)
    
    for sec in tqdm(
        range(command_line_args.start, command_line_args.end),
        unit="sec",
        desc="progress",
    ):

        # the text for visualization
        vis.text = ""

        # get the ground truth configuration and the camera-side ground truth video
        gt_args = munchify(settings.ground_truths_config.to_dict())
        gt_args.update(
            {"input": command_line_args.input, "second": sec,}
        )

        lq_args = gt_args.copy()
        lq_args.qp = settings.accmpeg.high_quality
        lq_args.tag = 'mpeg'
        lq_video_name, lq_video_config = encode(lq_args)
        lq_video = read_video_to_tensor(lq_video_name)

        H, W = settings.input_shape

        mask_shape = [
            len(lq_video),
            1,
            H // settings.dds.tile_size,
            W // settings.dds.tile_size,
        ]
        mask = torch.zeros(mask_shape).float()

        for idx, frame in enumerate(
            tqdm(lq_video, unit="frame", desc="saliency")
        ):
            frame = frame.unsqueeze(0)
            accmpeg_mask = mask_generator(frame.cuda()).softmax(dim=1)[:, 1:2, :, :]
            mask[idx:idx+1, :, :, :] = accmpeg_mask.cpu()

        
        

        dds_args = gt_args.copy()
        dds_args.tag = "accmpeg"
        dds_args.command_line_args = vars(command_line_args)
        del dds_args.qp
        mask = mask.mean(dim=0)[0]
        mask = (mask > settings.accmpeg.threshold).int()
        dds_args.macroblocks = mask * settings.accmpeg.high_quality + (1-mask) * settings.accmpeg.low_quality
        stat = examine(dds_args, gt_args, app, db)
        yaml_result = {
            'bandwidth': stat['my_video_config']['bw'],
            'compute': len(stat['my_video_config']['encoded_frames']),
            'input': dds_args.input,
            'second': dds_args.second,
            'f1': stat['f1'],
        }
        with open(f'{settings.root_dir}/stats/{command_line_args.approach}.yaml', 'a') as f:
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
        "--app", type=str, help="The name of the model.", default="EfficientDet-d2",
    )

    parser.add_argument("--approach", type=str, required=True)

    args = parser.parse_args()

    main(args)
