import argparse
import logging
import pickle
from pathlib import Path
from pdb import set_trace

import coloredlogs
import enlighten
import networkx as nx
import torch
import yaml
from torchvision import io
from munch import *
from dnn.dnn_factory import DNN_Factory
import numpy as np


# from utils.bbox_utils import jaccard
from inference import inference
import pickle
from datetime import datetime

default_config = {
    'confidence_threshold': 0.2,
    'gt_confidence_threshold': 0.2,
    'iou_threshold': 0.5
}
default_config = munchify(default_config)

def transform(result, lengt):
    print("In transform!!")
    
    factor = (lengt + (len(result) - 1)) // len(result) 

    print(len(result))
    print(lengt)
    print(factor)

    new_result = {}
    for i in range(lengt):
        new_result[i] = result[i // factor]

    return new_result



def examine(x_args, gt_args, x_app, gt_app, db, force=False, config=default_config):

    assert isinstance(x_args, Munch)
    assert isinstance(gt_args, Munch)
    
    x_args = x_args.copy()
    gt_args = gt_args.copy()
    
    # assert x_args.app == gt_args.app

    logger = logging.getLogger("examine")
    handler = logging.NullHandler()
    logger.addHandler(handler)

    query = x_args.copy() 
    query.update({'ground_truth': gt_args})

    if force==False and db['stats'].find_one(query) is not None:
        for x in db['stats'].find(query).sort("_id", pymongo.DESCENDING):
            return munchify(x)
    
    x = inference(x_args, db, x_app)
    gt = inference(gt_args, db, gt_app)
    
    x_dict = pickle.loads(x['inference_result'])
    gt_dict = pickle.loads(gt['inference_result'])
    dict_bf_transform = x_dict 
    if len(x_dict) != len(gt_dict):
        x_dict = transform(x_dict, len(gt_dict))

    # set_trace()
    
     
    metrics =  x_app.calc_accuracy(x_dict, gt_dict, config)
    per_frame_sum_cs = {}
    for fid in dict_bf_transform.keys():
        per_frame_sum_cs[fid] = torch.sum(dict_bf_transform[fid]['instances'].scores).item()
    print(per_frame_sum_cs)
    per_frame_sum_cs_transform = {}
    for fid in x_dict.keys():
        per_frame_sum_cs_transform[fid] = torch.sum(x_dict[fid]['instances'].scores).item()
    print(per_frame_sum_cs_transform)
    # print("Standard deviation", np.std(list(per_frame_sum_cs.values())))
    del x['inference_result']
    del x['timestamp']
    x['timestamp'] = str(datetime.now())
    x.update(metrics)
    x.update(config)
    x.update({'ground_truth': gt_args})
    x.update({'std_sum_confidence_score': np.std(list(per_frame_sum_cs_transform.values()))})
    x.update({'std_sum_confidence_score_original': np.std(list(per_frame_sum_cs.values()))})
    db['stats'].insert_one(x)
    db['stats'].insert_one(gt)
    # per_frame_sum_cs.update({'second': args.second})
    return x



if __name__ == "__main__":
    from dnn.dnn_factory import DNN_Factory
    import pymongo
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--inputs",
        help="The video file names. The largest video file will be the ground truth.",
        # default=['youtube_driving/chicago/chicago_%d/video' % i for i in range(20)],
    )
    parser.add_argument(
        "-g",
        "--gt_input",
        help="The video file names. The largest video file will be the ground truth.",
        # default=['youtube_driving/chicago/chicago_%d/video' % i for i in range(20)],
    )
    parser.add_argument(
        '--app',
        type=str,
        default='COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
    )
    parser.add_argument(
        '--stats',
        type=str,
        default='stats_la_driving'
    )
    parser.add_argument(
        '--res',
        type=str,
        default='1280:720'
    )
    parser.add_argument(
        '--second',
        type=str,
        default=0
    )

    parser.add_argument(
        '--fr',
        type=str,
        default=10
    )
    parser.add_argument(
        '--qp',
        type=str,
        default=0
    )
    parser.add_argument(
        '--image_idx',
        type=str,
        default='0'
    )
    args = parser.parse_args()
    print(args.inputs)

    x_args = {
        'app': 'EfficientDet-d8',
        'input': 'videos/dashcam/dashcam_{}/part{}.mp4',
        'second': int(args.second),
        'qp': int(args.qp),
        'fr': int(args.fr),
        'res': args.res,
        'image_idx': args.image_idx,
        'resize': args.res != "1280:720"
    }


    gt_args = {
        'app': 'EfficientDet-d8',
        'input': 'videos/dashcam/dashcam_{}/part{}.mp4',
        'second': args.second,
        'qp': 0,
        'fr': 10,
        'image_idx': args.image_idx,
        'res': '1280:720',
    }


    x_args = munchify(x_args)
    gt_args = munchify(gt_args)
    x_app = DNN_Factory().get_model(x_args.app, load_model=True)
    gt_app = DNN_Factory().get_model(gt_args.app, load_model=True)

    db = pymongo.MongoClient("mongodb://localhost:27017/")["diff_efficient_det_final"]


    print(examine(x_args, gt_args, x_app, gt_app, db))

