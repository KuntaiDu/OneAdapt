

import argparse
import gc
import logging
import time
from pathlib import Path

import io
import coloredlogs
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import yaml
# from torchvision import io
import io
from subprocess import run
import time

from pdb import set_trace
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dnn.dnn_factory import DNN_Factory

from utils.video_reader import read_video, read_video_config, read_video_to_tensor
# from utils.results import write_results

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
import torch
import os

import logging
import utils.config_utils as conf
sns.set()


__all__ = ['encode', 'tile_mask']


def tile_mask(mask, tile_size):
    """
        The mask should be of shape [H, W]
        This function will tile each element into a small tile x tile matrix.
        Eg: 
        mask = [    1   2
                    3   4]
        tile_mask(mask, 2) will return
        ret =  [    1   1   2   2
                    1   1   2   2
                    3   3   4   4
                    3   3   4   4]      
    """
    assert isinstance(mask, torch.Tensor)
    t = tile_size
    mask = mask.unsqueeze(1).repeat(1, t, 1).view(-1, mask.shape[1])
    mask = mask.transpose(0, 1)
    mask = mask.unsqueeze(1).repeat(1, t, 1).view(-1, mask.shape[1])
    mask = mask.transpose(0, 1)
    return mask
    # return torch.cat(3 * [mask[None, None, :, :]], 1)



def normal_encoding(args, input_video, output_video):
    
    # either specify the frame rate or provide the png of each frame.
    assert hasattr(args, 'fr') ^ ('png' in input_video)
    assert hasattr(args, 'res')
    assert hasattr(args, 'qp') ^ hasattr(args, 'macroblocks') 
    logger = logging.getLogger('encode')
    
    
    
    ffmpeg_command = ["ffmpeg"]
    ffmpeg_env = os.environ.copy()
    x264_dir = settings.x264_dir
    
    
    if hasattr(args, 'macroblocks'):
        
        
        
        
        
        # write QP matrix
        mask = args.macroblocks
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().numpy()
        else:
            assert isinstance(mask, list)
            mask = np.array(mask)
        logger.info(f'Perform RoI encoding with mean qp {mask.mean()}')
        # mask = torch.Tensor(mask)
        # assert len(mask.shape) == 2
        # assert settings.backprop.tile_size % 16 == 0
        # mask = tile_mask(mask, (settings.backprop.tile_size // 16))
        # mask =  mask.unsqueeze(0)
        
        # # binarize and the mask by 1
        # mask = mask.unsqueeze(0)
        # mask = (mask>0.5).float()
        # logger.info('Mean before padding: %.3f', mask.float().mean())
        # kernel = torch.ones([1, 1, 3, 3])
        # mask = F.conv2d(mask, kernel, stride=1, padding=1)
        # mask = (mask>0.5).int()
        # mask = mask[0]
        # logger.info('Mean after padding: %.3f', mask.float().mean())
        
        # mask = torch.cat([mask for i in range(10)])
        # mask = torch.where(
        #     mask > 0.5, 
        #     settings.backprop.high_quality * torch.ones_like(mask).int(),
        #     settings.backprop.low_quality * torch.ones_like(mask).int())
        
        
        bio = io.BytesIO()
        # for i in range(mask.shape[0]):
        # force the RoI to be the same
        np.savetxt(bio, mask, fmt="%i")
        qp_file_string = bio.getvalue().decode('latin1')

        # set_trace()
            


        # for i in range(mask.shape[0]):
        #     for j in range(mask.shape[1]):
        #         for k in range(mask.shape[2]):
        #             temp = mask[i,j,k]
        #             if temp > 0.5:
        #                 temp = settings.backprop.high_quality
        #             else:
        #                 temp = settings.backprop.low_quality
        #             qp_file_string += f"{temp} "
        #         qp_file += "\n"


        with open(f"{x264_dir}/qp_matrix_file", "w") as qp_file:

            qp_file.write(qp_file_string * 10)
                    
                    
        ffmpeg_command = [f"{x264_dir}/ffmpeg-3.4.8/ffmpeg"]
        ffmpeg_env["LD_LIBRARY_PATH"] = f"{x264_dir}/lib"
                    
        
        
    
    ffmpeg_command += [
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        # "-stats",
        "-i",
        input_video,
        "-s",
        f"{args.res}"
    ]
    
    if hasattr(args, 'bf'):
        ffmpeg_command += ["-bf", f"{args.bf}"]
    if hasattr(args, 'me_range'):
        ffmpeg_command += ["-me_range", f"{args.me_range}"]
    if hasattr(args, "me_method"):
        ffmpeg_command += ["-me_method", f"{args.me_method}"]
    if hasattr(args, "subq"):
        ffmpeg_command += ["-subq", f"{args.subq}"]
    if hasattr(args, 'fr'):
        ffmpeg_command += [
            "-filter:v",
            f"fps={args.fr}"]
    if 'qp' in args:
        ffmpeg_command += [
            '-c:v', 
            'libx264', 
            "-qp",
            f"{args.qp}"]
    ffmpeg_command += [output_video,]
    
    logger.info(f'Encoding command is: {ffmpeg_command}')
        

    subprocess.check_output(ffmpeg_command, env=ffmpeg_env)
        
    # input_video_config = read_video_config(input_video)
    output_video_config = read_video_config(output_video)
    output_video_config['encoded_frames'] = list(range(output_video_config['#frames']))
    
    return output_video_config



def encode(args):

    input_video = args.input % args.second
    # nput_video = args.input % 1
    prefix = f"cache/temp_{time.time()}"
    output_video = prefix + '.mp4'
    
    
    has_reducto = False
    for differencer in reducto_differencers:
        if hasattr(args, 'reducto_' + differencer.feature):
            has_reducto = True
            break
    
    if not has_reducto:
        
        return output_video, normal_encoding(args, input_video, output_video)


        
        
    else:

        assert args.fr == settings.ground_truths_config.fr, 'The frame rate of reducto must align with the ground truth.'

        logger = logging.getLogger('encode')
        logger.debug('Calculating frame differences')

        logger.info(f'Encode with {args}')


        # directly read input video and calculate reducto features.
        Path(prefix).mkdir()
        video_name = input_video
        # video_config = read_video_config(video_name)
        prev_frame = None
        prev_frame_pil = None
        
        remaining_frames = set()

        with ThreadPoolExecutor(max_workers=3) as executor:

            # for fid, frame in tqdm(read_video(video_name), total=video_config['#frames']):
            for fid_minus_one, frame in read_video(video_name):
                
                # do this because ffmpeg encodes from the 1st frame not the 0th frame.
                fid = fid_minus_one + 1
                
                # convert image to cv2 format for reducto
                cur_frame = np.array(T.ToPILImage()(frame[0]))[:, :, ::-1].copy()

                if prev_frame is None:
                    logger.debug('Encode frame %d', fid)

                    executor.submit(T.ToPILImage()(frame[0]).save, (prefix + '/%010d.png' % fid))
                    prev_frame = cur_frame
                    prev_frame_pil = T.ToPILImage()(frame[0])
                    
                    remaining_frames = remaining_frames | {fid}
                    
                else:

                    feature2values = {}
                    feature2weight = {}
                    feature2bias = {}
                    
                    for differencer in reducto_differencers:
                        if hasattr(args, 'reducto_' + differencer.feature + '_bias'):
                            difference_value = differencer.cal_frame_diff(differencer.get_frame_feature(cur_frame), differencer.get_frame_feature(prev_frame))
                            feature2values[differencer.feature] = difference_value
                            feature2weight[differencer.feature] = args['reducto_' + differencer.feature + '_weight']
                            feature2bias[differencer.feature] = args['reducto_' + differencer.feature + '_bias']
                            
                    weight = sum([feature2weight[key] * (feature2values[key] - feature2bias[key])]).sigmoid()
                    
                    if weight > 0.5:
                        logger.debug('Encode frame %d', fid)
                        executor.submit(T.ToPILImage()(frame[0]).save, (prefix + '/%010d.png' % fid))
                        prev_frame = cur_frame
                        prev_frame_pil = T.ToPILImage()(frame[0])
                        remaining_frames = remaining_frames | {fid}
                    else:
                        # Do not send the new frame
                        executor.submit(prev_frame_pil.save, (prefix + '/%010d.png' % fid))
                        

                
        logger.debug('%d frames are left after filtering, but still encode 10 frames to align the inference results.' % len(remaining_frames))
        
        output_video_config = normal_encoding(args, prefix + '/%010d.png', output_video)
        output_video_config['encoded_frames'] = list(remaining_frames)



        
        
        # subprocess.run(
        #     [
        #         "ffmpeg",
        #         "-y",
        #         "-hide_banner",
        #         "-loglevel",
        #         "error",
        #         # "-stats",
        #         "-i",
        #         prefix + '/%010d.png',
        #         "-s",
        #         f"{args.res}",
        #         '-c:v', 
        #         'libx264', 
        #         "-qp",
        #         f"{args.qp}",
        #         "-preset",
        #         f"{args.preset}",
        #         output_video,
        #     ]
        # )

        # shutil.rmtree(prefix)

        # output_video_config = read_video_config(output_video)
        # # video_config['#encoded_frames'] = set(range(video_config['#frames']))
        # output_video_config['encoded_frames'] = list(remaining_frames)
        




    return output_video, output_video_config


    
