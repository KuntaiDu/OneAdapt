import hashlib
import logging
import pickle
from collections import defaultdict
from itertools import product

import numpy as np
import torch
import torchvision.transforms as T
from joblib import Parallel, delayed

import utils.video_visualizer as vis
from utils.serialize import serialize_db_argument
from reducto.differencer import reducto_differencers
from utils.db_utils import find_in_collection
from utils.hash import sha256_hash

logger = logging.getLogger("reducto")


reducto_feature2meanstd = defaultdict(lambda: {"mean": 0.0, "std": 1.0})


def torchify(x):
    if isinstance(x, float):
        x = torch.tensor(x)
    return x


def calc_reducto_diff(cur_frame, prev_frame, args, is_pil_image, binarize):

    if not is_pil_image:
        cur_frame = np.array(T.ToPILImage()(cur_frame.detach()))[:, :, ::-1].copy()
        prev_frame = np.array(T.ToPILImage()(prev_frame.detach()))[:, :, ::-1].copy()

    feature2values = {}

    if feature2values == {}:
        for differencer in reducto_differencers:
            if hasattr(args, "reducto_" + differencer.feature + "_bias"):
                if differencer.feature not in feature2values:
                    difference_value = differencer.cal_frame_diff(
                        differencer.get_frame_feature(cur_frame),
                        differencer.get_frame_feature(prev_frame),
                    )
                    feature2values[differencer.feature] = difference_value
    # encode this frame only when all differences are greater than the thresholds
    weight = min(
        [
            torchify(
                (
                    (feature2values[key] - reducto_feature2meanstd[key]["mean"])
                    / reducto_feature2meanstd[key]["std"]
                )
                - args[f"reducto_{key}_bias"]
            ).sigmoid()
            for key in feature2values.keys()
        ]
    )

    if binarize:
        weight = (weight > 0.5).float()

    return weight, feature2values


# def reducto_process_on_frame(
#     video,
#     args,
#     cache,
#     cur_id,
#     prev_id,
#     prob,
#     prob_matrix,
#     compute,
#     metrics,
#     binarize=True,
# ):

#     if prob == 0:
#         # a not possible branch. return.
#         return

#     if cur_id >= len(video):

#         metrics["compute"] = metrics["compute"] + prob * compute
#         if prob > 0:
#             metrics["entropy"] = metrics["entropy"] - (prob * prob.log())

#     else:

#         if (cur_id, prev_id) not in cache:
#             cache[(cur_id, prev_id)] = calc_reducto_diff(
#                 video[cur_id],
#                 video[prev_id],
#                 args,
#                 is_pil_image=False,
#                 binarize=binarize,
#             )
#         new_prob = cache[(cur_id, prev_id)][0]
#         # if cur_id == prev_id + 1:
#         #     logger.info(f'{new_prob}')

#         # case 1: discard this frame w/ new_prob, the cur_id frame will be padded by prev_id frame.
#         # need to make sure this edit is not in-place, to preserve the gradient.
#         prob_matrix[cur_id, prev_id] = (
#             prob_matrix[cur_id, prev_id] + (1 - new_prob) * prob
#         )
#         reducto_process_on_frame(
#             video,
#             args,
#             cache,
#             cur_id + 1,
#             prev_id,
#             (1 - new_prob) * prob,
#             prob_matrix,
#             compute,
#             metrics,
#         )

#         # case 2: encode this frame w/ 1-new_prob
#         prob_matrix[cur_id, cur_id] = prob_matrix[cur_id, cur_id] + new_prob * prob
#         reducto_process_on_frame(
#             video,
#             args,
#             cache,
#             cur_id + 1,
#             cur_id,
#             new_prob * prob,
#             prob_matrix,
#             compute + 1,
#             metrics,
#         )


# def reducto_process(video, state, binarize=True):

#     cache = {}
#     metrics = {"compute": torch.tensor([0.0]), "entropy": torch.tensor([0.0])}
#     prob_matrix = defaultdict(lambda: 0.0)
#     for i in range(len(video)):
#         for j in range(len(video)):
#             prob_matrix[i, j] = torch.tensor([0.0])
#     prob_matrix[0, 0] = 1.0
#     compute = reducto_process_on_frame(
#         video,
#         state,
#         cache,
#         1,
#         0,
#         torch.tensor([1.0]),
#         prob_matrix,
#         1,
#         metrics,
#         binarize=binarize,
#     )

#     new_video = [torch.zeros_like(frame.unsqueeze(0)) for frame in video]

#     for cur_id in range(len(new_video)):
#         for prev_id in range(len(video)):
#             if prob_matrix[cur_id, prev_id] > 0:
#                 # need to make sure this edit is not in-place, to preserve the gradient.
#                 new_video[cur_id] = new_video[cur_id] + prob_matrix[
#                     cur_id, prev_id
#                 ] * video[prev_id].unsqueeze(0)
#     metrics["prob_matrix"] = prob_matrix

#     return torch.cat(new_video), metrics


def reducto_process(video, state, video_args, db):
    """ process the video with the reducto thresholds

    Args:
        video (torch.Tensor): the raw/ground-truth frames captured by the camera
        state (dict): The state dictionary
        video_args (dict): The args that derives the video parameter
        db (mongodb): The database pointer

    Returns:
        list: the list of encoded fids
    """
    
    query = {
        'video_args': serialize_db_argument(video_args.copy()),
        'state': serialize_db_argument(state.copy())
    }

    ret = find_in_collection(query, db['reducto_encode_fids'], 'reducto_process')
    
    if ret is None:
    
        # the first frame will always be encoded.
        encode_sequence = [0]
        prev_frame = None
        prev_fid = 0

        for fid, frame in enumerate(video):
            cur_frame = np.array(T.ToPILImage()(frame))[:, :, ::-1].copy()
            if fid == 0:
                prev_frame = cur_frame
                continue
            weight = calc_reducto_diff(
                cur_frame, prev_frame, state, is_pil_image=True, binarize=True
            )[0]

            if weight == 1:
                # Encode this frame
                prev_frame = cur_frame
                prev_fid = fid

            encode_sequence.append(prev_fid)
            
        ret = query
        ret['reducto_encode_fids'] = encode_sequence
        
        db['reducto_encode_fids'].insert_one(ret)
        
        
    # generate the processed video according to encode sequence
    ret_video = torch.cat([
        video[fid].unsqueeze(0)
        for fid in ret['reducto_encode_fids']
    ])

    return ret_video, ret['reducto_encode_fids']


def reducto_update_mean_std(video, state, gt_args, db):
    """ Get the mean and the standard deviation of all possible difference values
        on one video segment.

    Args:
        video (tensor): video encoded by gt_args
        state (dict): The state dict.
        gt_args (dict): the args for ground truth video
        db (mongodb): the database
    """

    cache = {}

    # def helper(i, j):
    #     if i < j:
    #         cache[i, j] = calc_reducto_diff(
    #             video[i], video[j], state, is_pil_image=False, binarize=True
    #         )

    # Parallel(n_jobs=4)(
    #     delayed(helper)(i, j) for i, j in product(range(len(video)), range(len(video)))
    # )

    query = gt_args.copy()
    query['gt_args_hash'] = sha256_hash(gt_args)
    
    ret = find_in_collection(query, db['reducto_mean_std'], 'reducto_mean_std')

    if ret is None:

        for i in range(len(video)):
            for j in range(i + 1, len(video)):
                cache[i, j] = calc_reducto_diff(
                    video[i], video[j], state, is_pil_image=False, binarize=True
                )

        feature2values = defaultdict(list)
        for frame_pairs in cache:
            feature2value = cache[frame_pairs][1]
            for feature in feature2value:
                feature2values[feature].append(feature2value[feature])

        for feature in feature2values:

            values_tensor = torch.tensor(feature2values[feature])
            mean, std = values_tensor.mean().item(), values_tensor.std().item()

            reducto_feature2meanstd[feature] = {"mean": mean, "std": std}

            logger.info("%s: mean %.3f, std: %.3f", feature, mean, std)

            vis.text += "%s: mean %.3f, std %.3f\n" % (feature, mean, std)

        ret = query
        ret['reducto_meanstd'] = reducto_feature2meanstd
        db['reducto_mean_std'].insert_one(ret)

    else:

        for feature in ret['reducto_meanstd'].keys():

            mean = ret['reducto_meanstd'][feature]['mean']
            std = ret['reducto_meanstd'][feature]['std']
    
            reducto_feature2meanstd[feature] = {"mean": mean, "std": std}

            logger.info("%s: mean %.3f, std: %.3f", feature, mean, std)

            vis.text += "%s: mean %.3f, std %.3f\n" % (feature, mean, std)

