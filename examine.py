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
import pymongo

from dnn.dnn_factory import DNN_Factory


# from utils.bbox_utils import jaccard
from inference import inference
import pickle
from datetime import datetime


def transform(result, lengt):

    
    factor = (lengt + (len(result) - 1)) // len(result)

    print(len(result))
    print(lengt)
    print(factor)

    new_result = {}
    for i in range(lengt):
        new_result[i] = result[i // factor]

    return new_result



def examine(x_args, gt_args, x_app, db, force=False):

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
    gt = inference(gt_args, db)
    
    x_dict = pickle.loads(x['inference_result'])
    gt_dict = pickle.loads(gt['inference_result'])

    if len(x_dict) != len(gt_dict):
        x_dict = transform(x_dict, len(gt_dict))

    # set_trace()
    
     
    metrics =  x_app.calc_accuracy(x_dict, gt_dict)
    
    del x['inference_result']
    del x['timestamp']

    x['timestamp'] = str(datetime.now())
    x.update(metrics)
    x.update({'ground_truth': gt_args})
    x.update({'norm_bw': x['bw'] * 1.0 / gt['bw']})

    db['stats'].insert_one(x)
    
    return x



if __name__ == "__main__":
    from dnn.dnn_factory import DNN_Factory
    import pymongo

    x_args = {
        'app': 'EfficientDet-d0',
        'input': 'videos/dashcam/dashcam_3/part%d.mp4',
        'second': 0,
        'qp': 0,
        'fr': 10,
        'res': '1280:720',
    }


    gt_args = {
        'app': 'EfficientDet-d8',
        'input': 'videos/dashcam/dashcam_3/part%d.mp4',
        'second': 0,
        'qp': 0,
        'fr': 10,
        'res': '1280:720',
    }

    x_args = munchify(x_args)
    gt_args = munchify(gt_args)
    x_app = DNN_Factory().get_model(x_args.app, load_model=False)


    db = pymongo.MongoClient("mongodb://localhost:27017/")["test"]


    print(examine(x_args, gt_args, x_app, db))

