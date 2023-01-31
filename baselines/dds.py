"""
    Compress the video through gradient-based optimization.
"""



import sys
from dynaconf import Dynaconf
import os
settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=[os.environ['SETTINGS_FILE']],
)
sys.path.append(settings.root_dir)
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import gc
import logging
import pickle
import random
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from pdb import set_trace
from typing import Tuple

import coloredlogs
import enlighten
import matplotlib.pyplot as plt
import pymongo
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import yaml
from detectron2.structures.instances import Instances
from munch import *
from PIL import Image, ImageDraw, ImageFont
from torch.utils.tensorboard import SummaryWriter
from torchvision import io
from tqdm import tqdm

import utils.config_utils as conf
import utils.video_visualizer as vis
from config import settings
from dnn.dnn import DNN
from dnn.dnn_factory import DNN_Factory
from utils.encode import encode, tile_mask, generate_mask_from_regions
from utils.gradient import (grad, grad_macroblocks, grad_reducto_cheap,
                            grad_reducto_expensive)
from utils.inference import examine, inference
from utils.reducto import (reducto_feature2meanstd, reducto_process,
                           reducto_update_mean_std)
from utils.seed import set_seed
from utils.timer import Timer
from utils.tqdm_handler import TqdmLoggingHandler
# from utils.results import write_results
from utils.video_reader import (read_video, read_video_config,
                                read_video_to_tensor)
from utils.video_visualizer import VideoVisualizer
from utils.visualize_utils import visualize_heat_by_summarywriter
from detectron2.structures.boxes import pairwise_iou
from utils.bbox_utils import center_size

sns.set()
set_seed()


conf.space = munchify(settings.configuration_space.to_dict())
state = munchify({})

logger = logging.getLogger("Server")

conf.serialize_order = list(settings.backprop.tunable_config.keys())

# sanity check
for key in settings.backprop.frozen_config.keys():
    assert (
        key not in conf.serialize_order
    ), f"{key} cannot both present in tunable config and frozen config."




def main(command_line_args):

    # a bunch of initialization.

    torch.set_default_tensor_type(torch.FloatTensor)

    db = pymongo.MongoClient("mongodb://localhost:27017/")[settings.collection_name]

    app = DNN_Factory().get_model(settings.dds.app)

    conf_thresh = settings[app.name].confidence_threshold

    logger.info("Application: %s", app.name)
    logger.info("Input: %s", command_line_args.input)
    logger.info("Approach: %s", command_line_args.approach)

    # visualizer
    visualize_folder = Path('debug/' + command_line_args.input).parent
    visualize_folder.mkdir(exist_ok=True, parents=True)
    
    for sec in tqdm(
        range(command_line_args.start, command_line_args.end),
        unit="sec",
        desc="progress",
    ):

        # the text for visualization
        vis.text = ""

        # get the ground truth configuration and the camera-side ground truth video
        gt_args = munchify(settings.ground_truths_config.to_dict())
        gt_args.update(
            {"input": command_line_args.input, "second": sec,}
        )

        lq_args = gt_args.copy()
        lq_args.qp = settings.dds.low_quality
        lq_args.tag = 'mpeg'
        lq_video_name, lq_video_config = encode(lq_args)
        lq_video = read_video_to_tensor(lq_video_name)

        H, W = settings.input_shape

        mask_shape = [
            len(lq_video),
            1,
            H // settings.dds.tile_size,
            W // settings.dds.tile_size,
        ]
        mask = torch.zeros(mask_shape).float()

        for idx, frame in enumerate(
            tqdm(lq_video, unit="frame", desc="saliency")
        ):
            frame = frame.unsqueeze(0)
            lq_inference = app.inference(frame, detach=True)
            lq_inference = app.filter_result(lq_inference)
            boxes = center_size(lq_inference["instances"].pred_boxes.tensor).cpu()
            maskA = generate_mask_from_regions(
                mask[idx:idx+1].clone(), boxes, 0, settings.dds.tile_size
            )

            proposals = app.region_proposal(frame, detach=True)
            proposals = proposals[proposals.objectness_logits > settings.dds.region_proposal_threshold]
            proposals = proposals[
                proposals.proposal_boxes.area() < settings.dds.size_threshold * H * W
            ]

            iou = pairwise_iou(
                proposals.proposal_boxes, lq_inference["instances"].pred_boxes
            )
            iou = iou > 0.3
            iou = iou.sum(dim=1)
            proposals = proposals[iou == 0]
            regions = center_size(proposals.proposal_boxes.tensor).cpu()

            maskB = generate_mask_from_regions(
                mask[idx:idx+1].clone(), regions, 0, settings.dds.tile_size
            )
            mask_delta = maskB - maskA
            mask_delta[mask_delta < 0] = 0
            mask[idx:idx+1, :, :, :] = mask_delta

        dds_args = gt_args.copy()
        dds_args.tag = "dds"
        dds_args.command_line_args = vars(command_line_args)
        del dds_args.qp
        mask = mask.sum(dim=[0,1])
        mask = (mask>0.5).int()
        dds_args.macroblocks = mask * settings.dds.high_quality + (1-mask) * settings.dds.low_quality
        examine(dds_args, gt_args, app, db)


        # stat, args, server_video = read_expensive_from_config(
        #     gt_args, state, app, db, command_line_args, train_flag
        # )
        # my_video_config = stat["my_video_config"]
        # logger.info("Actual compute: %d" % my_video_config["compute"])
        # vis.text += ("Comp: %d\n" "Acc : %.3f\n" "Bw  : %.3f\n") % (
        #     my_video_config["compute"],
        #     stat["acc"],
        #     stat["norm_bw"],
        # )
        # gt_results = pickle.loads(stat["gt_inference_result"])
        # my_results = pickle.loads(stat["my_inference_result"])

        # if train_flag:

        #     saliencies = {}
        #     raw_saliencies_tensor = []
        #     saliencies_tensor = []
        #     inference_results = {}
        #     camera_video = server_video
        #     compute = None
        #     reducto_encode_fids = None

        #     if (
        #         any("reducto" in key for key in state.keys())
        #         and not settings.backprop.reducto_expensive_optimize
        #     ):
        #         # get the differentiable reducto processed video on the camera
        #         camera_video, reducto_encode_fids = reducto_process(gt_video, state, gt_args, db)
        #         # new_vis.text = 'Estimated compute: %.3f' % metrics['compute'].item()
        #         # vis.text += new_vis.text
        #         # logger.info(new_vis.text)

        #     # calculate saliency on each frame.
        #     if command_line_args.loss_type == "saliency_error":
        #         for idx, frame in enumerate(
        #             tqdm(server_video, unit="frame", desc="saliency")
        #         ):

        #             if reducto_encode_fids is not None and idx not in reducto_encode_fids:
        #                 inference_results[idx] = inference_results[idx-1]
        #                 raw_saliencies_tensor.append(raw_saliencies_tensor[idx-1])
        #                 saliencies[idx] = saliencies[idx-1]
        #                 saliencies_tensor.append(saliencies_tensor[idx-1])
        #                 continue
        #             gt_frame = gt_video[idx]
        #             if settings.backprop.tunable_config.get("cloudseg", None):
        #                 with torch.no_grad():
        #                     frame = conf.SR_dnn(frame.unsqueeze(0).to("cuda:1")).cpu()[
        #                         0
        #                     ]
        #             frame = frame.unsqueeze(0)

        #             frame_detached = frame.detach()
        #             frame_detached.requires_grad_()
        #             result = app.inference(frame_detached, detach=False, grad=True)
        #             # filter out unrelated classes
        #             result = app.filter_result(result, confidence_check=False)
        #             score = result["instances"].scores
        #             if settings.backprop.saliency_type == "sigmoid":
        #                 sum_score = ((score - conf_thresh) * 20).sigmoid().sum()
        #             elif settings.backprop.saliency_type == "sum":
        #                 sum_score = score[score > conf_thresh].sum()
        #             # score_inds = (conf_lb < score) & (score < conf_ub)
        #             # logger.info('%d objects need for optimization.' % score_inds.sum())
        #             # score = score[score_inds]
        #             # sum_score = torch.sum(score)
        #             inference_results[idx] = result
        #             sum_score.backward()

        #             saliency = frame_detached.grad

        #             raw_saliencies_tensor.append(saliency.clone())
        #             saliency = saliency.abs().sum(dim=1, keepdim=True)

        #             # average across 16x16 neighbors
        #             kernel = torch.ones(
        #                 [
        #                     1,
        #                     1,
        #                     command_line_args.average_window_size,
        #                     command_line_args.average_window_size,
        #                 ]
        #             )
        #             saliency = F.conv2d(
        #                 saliency,
        #                 kernel,
        #                 stride=1,
        #                 padding=(command_line_args.average_window_size - 1) // 2,
        #             )
        #             saliencies[idx] = saliency
        #             saliencies_tensor.append(saliency)

        #     # video.requires_grad_()
        #     for iteration in range(command_line_args.num_iterations):

        #         # backprop on differentiable knobs.
        #         for idx, frame in enumerate(
        #             tqdm(server_video, unit="frame", desc="optimize")
        #         ):

        #             # bypass saliency calculation if performs expensive update
        #             if settings.backprop.skip_saliency_differentiable_backprop:
        #                 continue

        #             fid = sec * 10 + idx

        #             if settings.backprop.tunable_config.get("cloudseg", None):
        #                 frame = conf.SR_dnn(frame.unsqueeze(0).to("cuda:1")).cpu()[0]

        #             frame = frame.unsqueeze(0)

        #             reconstruction_loss = None
        #             result = None

        #             if command_line_args.loss_type == "saliency_error":

        #                 # result = app.inference(frame.detach(), detach=True, grad=False)
        #                 reconstruction_loss = (
        #                     saliencies[idx]
        #                     * (
        #                         gt_video[idx].unsqueeze(0)
        #                         - camera_video[idx].unsqueeze(0)
        #                     ).abs()
        #                 ).mean()
        #                 if frame.grad_fn is not None:
        #                     # frame is generated by differentiable knobs. Backward.
        #                     reconstruction_loss.backward(retain_graph=True)

        #             elif command_line_args.loss_type == "feature_error":

        #                 gt_result = app.inference(
        #                     gt_frame, detach=False, grad=False, feature=True
        #                 )
        #                 frame.retain_grad()
        #                 result = app.inference(
        #                     frame, detach=False, grad=True, feature=True
        #                 )

        #                 feature_diffs = []
        #                 for i in range(5):
        #                     feature_diffs.append(
        #                         (gt_result["features"][i] - result["features"][i])
        #                         .abs()
        #                         .mean()
        #                     )

        #                 reconstruction_loss = sum(feature_diffs)
        #                 reconstruction_loss.backward()
        #                 del result["features"]
        #                 del gt_result["features"]

        #                 saliency = frame.grad.abs().sum(dim=1, keepdim=True)
        #                 kernel = torch.ones(
        #                     [
        #                         1,
        #                         1,
        #                         command_line_args.average_window_size,
        #                         command_line_args.average_window_size,
        #                     ]
        #                 )
        #                 saliency = F.conv2d(
        #                     saliency,
        #                     kernel,
        #                     stride=1,
        #                     padding=(command_line_args.average_window_size - 1) // 2,
        #                 )

        #             else:
        #                 raise NotImplementedError

        #         # # backward cost parameters
        #         # if "metrics" in locals():
        #         #     if metrics["compute"].grad_fn is not None:
        #         #         (
        #         #             settings.backprop.compute_weight * metrics["compute"]
        #         #         ).backward()

        #         reducto_optimized = False
        #         for key in settings.backprop.tunable_config.keys():
        #             if key == "cloudseg":
        #                 # already optimized when backwarding.
        #                 continue
        #             elif key == "macroblocks":
        #                 grad_macroblocks(
        #                     state,
        #                     args,
        #                     key,
        #                     torch.cat(raw_saliencies_tensor),
        #                     gt_video_config,
        #                     gt_video,
        #                 )
        #                 continue
        #             elif "reducto" in key:
        #                 if not reducto_optimized:
        #                     if settings.backprop.reducto_expensive_optimize:
        #                         grad_reducto_expensive(
        #                             state, gt_video, gt_results, app, gt_args, db
        #                         )
        #                     else:
        #                         grad_reducto_cheap(
        #                             state, gt_video, torch.cat(saliencies_tensor), gt_args, db
        #                         )

        #                     reducto_optimized = True

        #                 continue
        #             grad(
        #                 args,
        #                 key,
        #                 torch.cat(saliencies_tensor),
        #                 gt_video_config,
        #                 gt_video,
        #             )

        #         optimizer.step()
        #         optimizer.zero_grad()

        #         # round to [0,1]
        #         for key in conf.serialize_order:
        #             if key == "cloudseg":
        #                 continue
        #             if key == "macroblocks":
        #                 continue
        #             if "reducto" in key:
        #                 continue
        #             state[key].requires_grad = False
        #             if state[key] > 1.0:
        #                 state[key][()] = 1.0
        #             if state[key] < 1e-7:
        #                 state[key][()] = 1e-7
        #             state[key].requires_grad = True

        #         # print out current state
                
        #         state_str = ""
        #         for key in conf.serialize_order:
        #             if key == "cloudseg":
        #                 param = list(conf.SR_dnn.net.parameters())[0]
        #                 logger.info(
        #                     "CloudSeg: gradient of layer 0 mean: %.3f, std: %.3f",
        #                     param.grad.mean(),
        #                     param.grad.std(),
        #                 )
        #                 continue
        #             if key == "macroblocks":
        #                 param = state[key]
        #                 logger.info("Macroblocks: mean: %.3f", param.float().mean())
        #                 continue
        #             if "reducto" in key:
        #                 param = state[key]
        #                 logger.info(f"{key}: %.3f", param.float().mean())
        #                 continue
        #             state_str += "%s : %.3f, grad: %.7f\n" % (
        #                 key,
        #                 state[key],
        #                 state[key].grad,
        #             )

        #         logger.debug(f"Current state: {state}")

        # # visualize
        # for idx, frame in enumerate(tqdm(server_video, desc="visualize", unit="frame")):

        #     image = T.ToPILImage()(frame.clamp(0, 1))

        #     draw = ImageDraw.Draw(image)
        #     font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 24)
        #     # manually bold the text :-P
        #     draw.multiline_text((10, 10), vis.text, fill=(255, 0, 0), font=font)
        #     draw.multiline_text((11, 10), vis.text, fill=(255, 0, 0), font=font)
        #     draw.multiline_text((12, 10), vis.text, fill=(255, 0, 0), font=font)

        #     my_result = app.filter_result(my_results[idx])
        #     gt_result = app.filter_result(gt_results[idx], gt=True)

        #     (
        #         gt_ind,
        #         my_ind,
        #         gt_filtered,
        #         my_filtered,
        #     ) = app.get_undetected_ground_truth_index(my_result, gt_result)

        #     image_error = app.visualize(
        #         image,
        #         {
        #             "instances": Instances.cat(
        #                 [gt_filtered[gt_ind], my_filtered[my_ind]]
        #             )
        #         },
        #     )
        #     image_inference = app.visualize(image, my_result)
        #     errors_vis.add_frame(image_error)
        #     results_vis.add_frame(image_inference)

        # logger.info("Visualize text:\n%s", vis.text)


if __name__ == "__main__":

    # set the format of the logger
    formatter = coloredlogs.ColoredFormatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s",
        datefmt="%H:%M:%S",
    )

    handler = TqdmLoggingHandler()
    handler.setFormatter(formatter)
    logging.basicConfig(handlers=[handler], level="INFO")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input", help="The format of input video.", type=str, required=True
    )

    parser.add_argument(
        "--start", help="The total secs of the video.", required=True, type=int
    )

    parser.add_argument(
        "--end", help="The total secs of the video.", required=True, type=int
    )

    parser.add_argument(
        "--app", type=str, help="The name of the model.", default="EfficientDet-d2",
    )

    parser.add_argument("--approach", type=str, required=True)

    args = parser.parse_args()

    main(args)
