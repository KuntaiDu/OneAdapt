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

from dnn.dnn_factory import DNN_Factory
import pymongo

from config import settings

# gt_qp = 24
# qp_list = [24]
# qp_list = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]

# qp_list = [24]
# fr_list = [30]
# res_list = {720:"1280:720"}

# qp_list = [24, 26, 30, 32, 34, 36, 40, 44, 50]
# fr_list = [30, 10, 5, 3]
# res_list = {240:"352:240", 360:"480:360", 480:"858:480", 720:"1280:720"}


# gt = 'qp_24_fr_30_res_720'

gt_config = munchify(settings.ground_truths_config)



video_name = settings.video_name
total_sec = settings.num_segments
db = pymongo.MongoClient("mongodb://localhost:27017/")[settings.collection_name]

# settings.inference_config.enable_visualization = True


if __name__ == "__main__":

    logger = logging.getLogger("mpeg_curve")

    app = DNN_Factory().get_model(gt_config.app)

    for sec in range(total_sec):

        for gamma in [1.0]:

            args = gt_config.copy()
            args.update({
                'input': video_name,
                'second': sec,
                'gamma': gamma
            })

            inference(args, db, app)

        


