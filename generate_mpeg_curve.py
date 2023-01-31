import argparse
import logging
import os
os.environ['SETTINGS_FILE'] = 'settings_reducto.toml'
import subprocess
from pathlib import Path

from munch import *
from pdb import set_trace

from itertools import product
import coloredlogs

from utils.inference import inference, examine

from dnn.dnn_factory import DNN_Factory
import pymongo
from tqdm import tqdm
import yaml


from config import settings





gt_config = munchify(settings.ground_truths_config.to_dict())



def probe_range(fmt):
    
    idx = 0
    while Path(fmt % idx).exists():
        idx += 1
    return idx

fmts = [
    f'/dataheart/dataset/rural/rural_{i}/part%d.mp4' for i in range(11)
]
# fmts = [
#     f"/dataheart/dataset/country/country_{i}/part%d.mp4" for i in range(6)
# ]

# fmts = [
#     f'videos/driving/driving_{i}/part%d.mp4' for i in [2]
# ]

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
    # for pixel_thresh in settings.configuration_space.reducto_pixel[::-1]:
    #     for area_thresh in settings.configuration_space.reducto_area:
    #         for edge_thresh in settings.configuration_space.reducto_edge:

    for idx, fmt in enumerate(fmts):

        if idx != 0:
            continue

        f1s = []

        app = DNN_Factory().get_model(gt_config.app)

        # for sec in range(probe_range(fmt)):
        for sec in tqdm(range(119)):

            gt_args = gt_config.copy()
            gt_args.update({
                'input': fmt,
                'second': sec,
                'approach': 'mpeg'
            })

            del gt_args.reducto_pixel_bias
            del gt_args.reducto_area_bias

            x_args = gt_args.copy()
            x_args.fr = 1

            # del x_args.fr
            # x_args.reducto_pixel = pixel_thresh
            # x_args.reducto_area = area_thresh
            # x_args.reducto_edge = edge_thresh

            x = examine(x_args,gt_args,app,db)
            f1s.append(x['f1'])
        
        with open('stats/rural_%d.f1s' % idx, 'w') as f:
            f.write(yaml.dump(f1s))


