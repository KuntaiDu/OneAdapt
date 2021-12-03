
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
from concurrent.futures import ThreadPoolExecutor
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
import numpy as np

from config import settings
from reducto.differencer import reducto_differencers
import shutil

import logging
sns.set()


__all__ = ['inference', 'encode']





def encode(args):

    input_video = args.input % args.second
    prefix = f"cache/temp_{time.time()}"
    output_video = prefix + '.mp4'
    
    has_reducto = False
    for differencer in reducto_differencers:
        if hasattr(args, 'reducto_' + differencer.feature):
            has_reducto = True
            break
    
    if not has_reducto:

        subprocess.check_output(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-stats",
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
        
        remaining_frames = None # no filtering there.
        
    else:

        assert not hasattr(args, 'fr'), 'Cannot use reducto while performing frame subsampling.'

        logger = logging.getLogger('encode')
        logger.info('Calculating frame differences')


        # directly read input video and calculate reducto features.
        Path(prefix).mkdir()
        video_name = input_video
        video_config = read_video_config(video_name)
        prev_frame = None
        prev_frame_pil = None
        
        remaining_frames = 0

        with ThreadPoolExecutor(max_workers=3) as executor:

            for fid, frame in tqdm(read_video(video_name), total=video_config['#frames']):
                
                # convert image to cv2 format for reducto
                cur_frame = np.array(T.ToPILImage()(frame[0]))[:, :, ::-1].copy()

                if prev_frame is None:
                    logger.info('Encode frame %d', fid)

                    executor.submit(T.ToPILImage()(frame[0]).save, (prefix + '/%010d.png' % fid))
                    prev_frame = cur_frame
                    prev_frame_pil = T.ToPILImage()(frame[0])
                    
                    remaining_frames = 1
                    
                else:

                    discard = True
                    
                    for differencer in reducto_differencers:
                        if hasattr(args, 'reducto_' + differencer.feature):
                            difference_value = differencer.cal_frame_diff(differencer.get_frame_feature(cur_frame), differencer.get_frame_feature(prev_frame))
                            logger.info('Frame %d, feat %s, value %.5f', fid, differencer.feature, difference_value)
                            if difference_value > getattr(args, 'reducto_' + differencer.feature):
                                discard = False
                                break

                    if not discard:
                        logger.info('Encode frame %d', fid)
                        executor.submit(T.ToPILImage()(frame[0]).save, (prefix + '/%010d.png' % fid))
                        prev_frame = cur_frame
                        prev_frame_pil = T.ToPILImage()(frame[0])
                        remaining_frames += 1
                    else:
                        executor.submit(prev_frame_pil.save, (prefix + '/%010d.png' % fid))

                
        logger.info('%d frames are left after filtering, but still encode 10 frames to align the inference results.' % remaining_frames)



        
        
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-stats",
                "-i",
                prefix + '/%010d.png',
                "-s",
                f"{args.res}",
                '-c:v', 
                'libx264', 
                "-qp",
                f"{args.qp}",
                output_video,
            ]
        )

        shutil.rmtree(prefix)

    return output_video, remaining_frames


    




def inference(args, db, app=None):

    logger = logging.getLogger('inference')
    config = settings.inference_config



    # logger.info('Inference on %s with res %s, fr %d, qp %d, app %s, gamma %.2f', args.input % args.second, args.res, args.fr, args.qp, args.app, args.gamma)
    args_string = ""
    for key in args:
        args_string += f'{key}_{args[key]}_'
    args = args.copy()
    logger.info('Try inference on %s', args_string)

    logger = logging.getLogger("inference")
    handler = logging.NullHandler()
    logger.addHandler(handler)

    torch.set_default_tensor_type(torch.FloatTensor)


    # check if we already performed inference.
    if db['inference'].find_one(args) is not None:

        if not config.force_inference:

            logger.info('Inference results already cached. Return.')
            for x in db['inference'].find(args).sort("_id", pymongo.DESCENDING):
                return munchify(x)

        else:
            
            logger.warning('Previous inference results exist, but force inference.')
            

    logger.info('Start inference.')

    # prepare for inference
    
    if config.enable_visualization:
        logger.info('Launch visualization')
        writer = SummaryWriter('runs/'+ args_string)
        

    assert app is not None and app.name == args.app
    video_name, remaining_frames = encode(args)
    video_config = read_video_config(video_name)




    # inference
    logger.info("Running %s", args_string)

    inference_results = {}

    for fid, frame in tqdm(read_video(video_name), total=video_config['#frames']):
            
        if hasattr(args, 'gamma'):
            frame = T.functional.adjust_gamma(frame, args.gamma)

        inference_results[fid] = app.inference(frame, grad=False, detach=True, dryrun=False)

        
            

        if config.enable_visualization and fid % config.visualize_step_size == 0:

            logger.info('Visualizing frame %d...', fid)
            image = T.ToPILImage()(frame[0])
            from PIL import Image

            writer.add_image("decoded_image", T.ToTensor()(image), fid)
            # filtered = inference_results[fid]
            # set_trace()
            filtered = app.filter_result(inference_results[fid])
            image = app.visualize(image, filtered)
            writer.add_image("inference_result", T.ToTensor()(image), fid)



    # update args and insert to database
    args.update({
        'inference_result': pickle.dumps(inference_results),
        'timestamp': str(datetime.now()),
        'remaining_frames': remaining_frames,
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
