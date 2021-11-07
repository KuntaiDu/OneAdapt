
import argparse
import gc
import logging
import time
from pathlib import Path

import coloredlogs
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import yaml
from torchvision import io
from subprocess import run
import time

from pdb import set_trace
from datetime import datetime

from dnn.dnn_factory import DNN_Factory

from utils.video_reader import read_video, read_video_config
from utils.results import write_results

from knob.control_knobs import framerate_control, quality_control
from tqdm import tqdm
from munch import *
from datetime import datetime
import pymongo
import subprocess
import pickle

sns.set()


default_size = (800, 1333)
default_config = Munch()
default_config.visualize = False
default_config.force = False




def encode(args):

    input_video = args.input % args.second
    output_video = f'temp_{time.time()}.mp4'


    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_video,
            "-s",
            f"{args.res}",
            "-filter:v",
            f"fps={args.fr}",
            '-c:v', 
            'libx264', 
            "-qp",
            f"{args.qp}",
            output_video,
        ]
    )

    return output_video


    




def inference(args, db, app=None, config=default_config):

    args = args.copy()

    logger = logging.getLogger("inference")
    handler = logging.NullHandler()
    logger.addHandler(handler)

    torch.set_default_tensor_type(torch.FloatTensor)


    # check if we already performed inference.
    if not config.force and db['inference'].find_one(args) is not None:
        for x in db['inference'].find(args).sort("_id", pymongo.DESCENDING):
            return munchify(x)


    # prepare for inference
    args_string = ""
    for key in args:
        args_string += f'{key}_{args[key]}'
    if config.visualize:
        writer = SummaryWriter(runs + args_string)

    assert app is not None and app.name == args.app
    video_name = encode(args)
    video_config = read_video_config(video_name)




    # inference
    logger.info("Running %s", args_string)

    inference_results = {}

    for fid, frame in tqdm(read_video(video_name), total=video_config['#frames']):

        if hasattr(args, 'resize') and args.resize:
            frame = F.interpolate(frame, size=default_size)

        inference_results[fid] = app.inference(frame, grad=False, detach=True, dryrun=False)
            

        if config.visualize and fid % config.visualize == 0:
            logger.info('Visualizing frame %d...', fid)
            image = T.ToPILImage()(frame[0])
            from PIL import Image

            writer.add_image("decoded_image", T.ToTensor()(image), fid)
            # filtered = inference_results[fid]
            # set_trace()
            filtered = app.filter_result(inference_results[fid], args)
            image = app.visualize(image, filtered, args)
            writer.add_image("inference_result", T.ToTensor()(image), fid)



    # update args and insert to database
    args.update({
        'inference_result': pickle.dumps(inference_results),
        'timestamp': str(datetime.now()),
    })
    args.update(video_config)
    # insert result to database
    db['inference'].insert_one(args)
    
    # cleanup
    Path(video_name).unlink()
    
    return args
# if __name__ == "__main__":

#     # set the format of the logger
#     coloredlogs.install(
#         fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s",
#         level="INFO",
#     )

#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "-i",
#         "--input",
#         type=str,
#         help="The video file names to obtain inference results.",
#         required=True,
#     )
    
#     parser.add_argument(
#         "--second",
#         type=int,
#         help="Running which second of the video.",
#         required=True
#     )
#     parser.add_argument(
#         "--app", type=str, help="The name of the model.", default='COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml',
#     )
    
#     parser.add_argument(
#         '--resize', default=False, action='store_true'
#     )

#     # visualization parameters
#     parser.add_argument(
#         '--visualize', default=False, action='store_true'
#     )
#     parser.add_argument(
#         '--visualize_step_size',
#         type=int,
#         default=100
#     )
#     parser.add_argument(
#         "--confidence_threshold",
#         type=float,
#         help="The confidence score threshold for calculating accuracy.",
#         default=0.5,
#     )
#     parser.add_argument(
#         "--gt_confidence_threshold",
#         type=float,
#         help="The confidence score threshold for calculating accuracy.",
#         default=0.5,
#     )
#     parser.add_argument(
#         "--iou_threshold",
#         type=float,
#         help="The IoU threshold for calculating accuracy in object detection.",
#         default=0.5,
#     )

#     args = parser.parse_args()

#     main(args)
