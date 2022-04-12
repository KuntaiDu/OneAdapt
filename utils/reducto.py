
from reducto.differencer import reducto_differencers
import torch
from collections import defaultdict
import numpy as np
import torchvision.transforms as T
import logging

logger = logging.getLogger("reducto")


reducto_feature2meanstd = {}



def calc_reducto_diff(cur_frame, prev_frame, args, is_pil_image):

    if not is_pil_image:
        cur_frame = np.array(T.ToPILImage()(cur_frame.detach()))[:, :, ::-1].copy()
        prev_frame = np.array(T.ToPILImage()(prev_frame.detach()))[:, :, ::-1].copy()

    feature2values = {}
    
    if feature2values == {}:
        for differencer in reducto_differencers:
            if hasattr(args, 'reducto_' + differencer.feature + '_bias'):
                if differencer.feature not in feature2values:
                    difference_value = differencer.cal_frame_diff(differencer.get_frame_feature(cur_frame), differencer.get_frame_feature(prev_frame))
                    feature2values[differencer.feature] = (difference_value - reducto_feature2meanstd[differencer.feature]['mean']) / reducto_feature2meanstd[differencer.feature]['std']
                    
    def torchify(x):
        if isinstance(x, float):
            x = torch.tensor(x)
        return x
    
    # encode this frame only when all differences are greater than the thresholds
    weight = min([
        torchify(feature2values[key] - args['reducto_' + key + '_bias']).sigmoid()
        for key in feature2values.keys()])

    return weight, feature2values


def reducto_process_on_frame(video, args, cache, cur_id, prev_id, prob, prob_matrix, compute, metrics):

    if cur_id >= len(video):

        metrics['compute'] = metrics['compute'] + prob * compute
        if prob > 0:
            metrics['entropy'] = metrics['entropy'] - (prob * prob.log())

    else:

        if (cur_id, prev_id) not in cache:
            cache[(cur_id, prev_id)] = calc_reducto_diff(video[cur_id], video[prev_id], args, is_pil_image=False)
        new_prob = cache[(cur_id, prev_id)][0]
        # if cur_id == prev_id + 1:
        #     logger.info(f'{new_prob}')


        # case 1: discard this frame w/ new_prob, the cur_id frame will be padded by prev_id frame.
        # need to make sure this edit is not in-place, to preserve the gradient.
        prob_matrix[cur_id, prev_id] = prob_matrix[cur_id, prev_id] + (1- new_prob) * prob
        reducto_process_on_frame(video, args, cache, cur_id + 1, prev_id, (1- new_prob) * prob, prob_matrix, compute, metrics)

        # case 2: encode this frame w/ 1-new_prob
        prob_matrix[cur_id,cur_id] = prob_matrix[cur_id, cur_id] + new_prob * prob
        reducto_process_on_frame(video, args, cache, cur_id + 1, cur_id, new_prob * prob, prob_matrix, compute + 1, metrics)







def reducto_process(video, state):

    cache = {}
    metrics = {'compute': torch.tensor([0.]), 'entropy': torch.tensor([0.])}
    prob_matrix = defaultdict(lambda: 0.0)
    for i in range(len(video)):
        for j in range(len(video)):
            prob_matrix[i, j] = torch.tensor([0.0])    
    prob_matrix[0, 0] = 1.0
    compute = reducto_process_on_frame(video, state, cache, 1, 0, torch.tensor([1.0]), prob_matrix, 1, metrics)

    new_video = [torch.zeros_like(frame.unsqueeze(0)) for frame in video]

    for cur_id in range(len(new_video)):
        for prev_id in range(len(video)):
            # need to make sure this edit is not in-place, to preserve the gradient.
            new_video[cur_id] = new_video[cur_id] + prob_matrix[cur_id, prev_id] * video[prev_id].unsqueeze(0)

    return torch.cat(new_video), metrics



def reducto_update_mean_std(video, state):

    cache = {}
    metrics = {'compute': torch.tensor([0.]), 'entropy': torch.tensor([0.])}
    prob_matrix = defaultdict(lambda: 0.0)
    for i in range(len(video)):
        for j in range(len(video)):
            prob_matrix[i, j] = torch.tensor([0.0])    
    prob_matrix[0, 0] = 1.0

    with torch.no_grad():
        
        reducto_process_on_frame(video, state, cache, 1, 0, torch.tensor([1.0]), prob_matrix, 1, metrics)

    feature2values = defaultdict(list)
    for frame_pairs in cache:
        feature2value = cache[frame_pairs][1]
        for feature in feature2value:
            feature2values[feature].append(feature2value[feature])

    for feature in feature2values:

        values_tensor = torch.tensor(feature2values[feature])
        mean, std = values_tensor.mean(), values_tensor.std()
        
        reducto_feature2meanstd[feature] = {'mean': mean, 'std': std}

