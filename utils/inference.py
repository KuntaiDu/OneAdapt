
import argparse
import gc
import logging
import pickle
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from pdb import set_trace
from subprocess import run

import coloredlogs
import matplotlib.pyplot as plt
import numpy as np
import pymongo
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from config import settings
from dnn.dnn_factory import DNN_Factory
from munch import *
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import io
from tqdm import tqdm

import utils.config_utils as conf
from reducto.differencer import reducto_differencers
from utils.db_utils import find_in_collection
from utils.encode import encode
from utils.hash import sha256_hash
from utils.serialize import serialize_db_argument
from utils.video_reader import (read_video, read_video_config,
                                read_video_to_tensor)

__all__ = ['inference', 'examine']

def transform(result, lengt):
    
    factor = (lengt + (len(result) - 1)) // len(result)

    print(len(result))
    print(lengt)
    print(factor)

    new_result = {}
    for i in range(lengt):
        new_result[i] = result[i // factor]

    return new_result



def examine(my_args, gt_args, my_app, db, key='examine', profile=False):

    assert isinstance(my_args, Munch)
    assert isinstance(gt_args, Munch)
    
    my_args = my_args.copy()
    gt_args = gt_args.copy()
    my_args = serialize_db_argument(my_args)
    gt_args = serialize_db_argument(gt_args)
    
    logger = logging.getLogger("examine")
    # add the hash to force the match of my_args and gt_args in the database must be exact
    query = {
        'my_args': my_args,
        'gt_args': gt_args,
    }
    
    
    ret = find_in_collection(query, db[key], 'examine', settings.examine_config.force_examine)
    if ret is None:
        
        my = inference(my_args, db, my_app)
        gt = inference(gt_args, db, my_app)  # will raise an error inside if the model of GT does not align with the model of x.
        
        
        my_dict = pickle.loads(my['inference_result'])
        gt_dict = pickle.loads(gt['inference_result'])

        if len(my_dict) != len(gt_dict):
            my_dict = transform(my_dict, len(gt_dict))
        
        ret = query
        ret['my_inference_result'] = my['inference_result']
        ret['gt_inference_result'] = gt['inference_result']
        ret['my_video_config'] = my['video_config']
        ret['gt_video_config'] = gt['video_config']
        # update accuracy metrics
        ret.update(my_app.calc_accuracy(my_dict, gt_dict))
        # update normalized bandwidth
        if profile:
            # need to stream the video from the camera to the server.
            my['video_config']['bw'] = gt['video_config']['bw']
            ret['my_video_config']['bw'] = gt['video_config']['bw']
            ret['f1'] = 1.0
        ret.update({'norm_bw': my['video_config']['bw'] * 1.0 / gt['video_config']['bw']})

        db[key].insert_one(ret)
    
    return munchify(ret)







def inference(args, db, app=None, video_name=None, video_config=None):

    logger = logging.getLogger('inference')

    config = settings.inference_config



    # logger.info('Inference on %s with res %s, fr %d, qp %d, app %s, gamma %.2f', args.input % args.second, args.res, args.fr, args.qp, args.app, args.gamma)
    args = args.copy()
    args = serialize_db_argument(args)
    args_string = ""
    for key in args:
        args_string += f'{key}_{args[key]}_'
    logger.debug("Encoding args %s", args_string)
    
    assert app is not None and app.name == args.app.replace('.', '_').replace('/', '_'), f'{args}'
    if video_name is not None:
        assert video_config is not None
    else:
        video_name, video_config = encode(args)
        
    # different encoding may result in the same encoded video.
    # So only index the inference results with its source, the video file hash and the DNN
    query = {
        'input': args.input,
        'second': args.second,
        'encoded_video_hash': video_config['sha256'],
        'app': args.app
    }

    ret = find_in_collection(query, db['inference'], 'inference', config.force_inference)
    
    if ret is None:
        # result not found. Run inference.

        inference_results = {}
        encoded_fids = video_config['encoded_frames']
        
        # inference.
        with torch.no_grad():
            for fid, frame in enumerate(tqdm(read_video_to_tensor(video_name), unit='frame', desc='inference')):
                if fid in encoded_fids:
                    # inference
                    frame = frame.unsqueeze(0)
                    inference_results[fid] = app.inference(frame, grad=False, detach=True, dryrun=False)
                else:
                    inference_results[fid] = deepcopy(inference_results[fid-1])
                
        ret = query
        ret.update({
            'inference_result': pickle.dumps(inference_results),
            'video_config': video_config
        })
        db['inference'].insert_one(ret)
        
        # cleanup
        Path(video_name).unlink()
    
    return munchify(ret)
