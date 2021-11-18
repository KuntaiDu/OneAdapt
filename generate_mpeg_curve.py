import argparse
import logging
import os
import subprocess
from pathlib import Path

from munch import *
from pdb import set_trace

from utils.results import read_results
from itertools import product

from inference import inference
from examine import examine

from dnn.dnn_factory import DNN_Factory
import pymongo


from config import settings



gt_config = settings.ground_truths_config
gt_config = munchify(gt_config)


video_name = settings.video_name
total_sec = settings.num_segments
db = pymongo.MongoClient("mongodb://localhost:27017/")[settings.collection_name]




if __name__ == "__main__":

    logger = logging.getLogger("mpeg_curve")

    # for app_name in app_name_list: 
    # for app_name in ['EfficientDet-d8']:
    # for gamma in [0.5, 0.6, 0.7, 0.8, 0.9]:
    for idx, gamma in enumerate([0.5, 0.6, 0.7, 0.8, 0.9, 1.01, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]):
        if idx % 3 != 2:
            continue

        app = DNN_Factory().get_model(gt_config.app)

        for sec in range(total_sec):

            gt_args = gt_config.copy()
            gt_args.update({
                'input': video_name,
                'second': sec
            })

            x_args = gt_args.copy()
            x_args.gamma = gamma 

            examine(x_args,gt_args,app,db)
        


