import argparse
import logging
import os
import subprocess
from pathlib import Path

from munch import *
from pdb import set_trace

from itertools import product
import coloredlogs

from inference import inference
from examine import examine

from dnn.dnn_factory import DNN_Factory
import pymongo


from config import settings




gt_config = munchify(settings.ground_truths_config.to_dict())


video_name = settings.video_name
total_sec = settings.num_segments
db = pymongo.MongoClient("mongodb://localhost:27017/")[settings.collection_name]




if __name__ == "__main__":

    coloredlogs.install(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s",
        level="INFO",
    )

    logger = logging.getLogger("mpeg_curve")

    # for app_name in app_name_list: 
    # for app_name in ['EfficientDet-d8']:
    # for gamma in [0.5, 0.6, 0.7, 0.8, 0.9]:
    # for pixel_thresh in [0.4,0.35, 0.3, 0.25, 0.2, 0.15]:
    # for area_thresh in [0.2, 0.15, 0.1, 0.05]:
    # for edge_thresh in [0.0075, 0.008, 0.0085, 0.009]:

    app = DNN_Factory().get_model(gt_config.app)

    for sec in range(total_sec):

        gt_args = gt_config.copy()
        gt_args.update({
            'input': video_name,
            'second': sec,
        })

        x_args = gt_args.copy()
        x_args.cloudseg = True
        x_args.res = '640:360'
        x_args.approach = 'CloudSeg'

        examine(x_args,gt_args,app,db)
        


