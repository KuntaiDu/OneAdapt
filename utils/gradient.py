'''
    Calculate gradient for each knob.
    May also optimize that knob if closed-form solution exists / in the baseline
'''

import logging

from utils.video_reader import read_video, read_video_config, read_video_to_tensor
import utils.config_utils as conf
from utils.reducto import reducto_process_on_frame, reducto_process, reducto_feature2meanstd
import torch
from config import settings
from utils.tqdm_handler import TqdmLoggingHandler
from itertools import product
from copy import deepcopy
from tqdm import tqdm
import utils.video_visualizer as vis


__all__ = ['grad_macroblocks', 'grad', 'grad_reducto_expensive', 'grad_reducto_cheap', 'grad']

logger = logging.getLogger("gradient")

def grad_macroblocks(args: dict, key: str, grad: torch.Tensor, gt_config, gt_video):

    args = args.copy()
    
    tile_size = settings.backprop.tile_size
    
    
    # generate ground truth
    gt_args = args.copy()
    del gt_args.macroblocks
    gt_args.qp = settings.ground_truths_config.qp
    logger.info(f'Getting ground truth video')
    gt_name, gt_config = encode(gt_args)
    gt_video = read_video_to_tensor(gt_name)
    
    
    
    
    # generate videos with different qp levels
    qps = list(settings.backprop.qps)
    bws, videos = [], []
    for qp in qps:
        
        qp_args = gt_args.copy()
        qp_args.qp = qp
        logger.info(f'Encode with QP {qp_args.qp}')
        qp_name, qp_config = encode(qp_args)
        qp_video = read_video_to_tensor(qp_name)
        
        bws.append(qp_config['bw'])
        videos.append(qp_video)
    
    # pixel error
    emaps = [abs(video - gt_video) for video in videos]
    
    # saliency * pixel error
    saliency_sums = [F.conv2d(
        abs(grad * emap).sum(dim=1, keepdim=True).sum(dim=0, keepdim=True), 
        torch.ones([1, 1, tile_size, tile_size]),
        stride=tile_size
    ) for emap in emaps]
    
    perc = settings.backprop.bw_percentage
    
    
    delta = (saliency_sums[1] - saliency_sums[0]).detach()
    delta = delta.flatten().sort()[0]
    delta = delta[-int(len(delta) * perc)]
    
    relative_bw = bws[1] / bws[0]
    bw_weight = delta / (1-relative_bw)
    objectives = [saliency_sums[i] + bw_weight * torch.ones_like(saliency_sums[i]) * bws[i] / bws[0] for i in range(len(qps))]
    
    # start from the lowest quality.
    qpmap = torch.ones_like(objectives[-1]).int() * qps[-1]
    current_objective = objectives[-1]
    for i in range(0, len(qps)):
        qpmap = torch.where(current_objective > objectives[i], torch.ones_like(qpmap) * qps[i], qpmap)
        current_objective = torch.where(current_objective > objectives[i], objectives[i], current_objective)
        
    # finalize optimization, prep 
    state[key] = qpmap[0,0]
    
    return True




def grad_reducto_expensive(state, gt_video, results, app):
    """Optimize reducto thresholds

    Args:
        state (dict): the state dict that includes reducto thresholds
        gt_video (torch.tensor): the ground-truth video
        results (dict): the ground truth inference results
        app: calculate the inference accuracy
    
    Returns:
        dict: the optimal objective function
    """  
    
    
    keys = list(reducto_feature2meanstd.keys())
    
    min_objective = {}
    min_state = {}
    
    for candidate in tqdm(list(product(*[[-2, 1,0, 1, 2] for key in keys])), desc='reducto expensive search'):
        for idx, val in enumerate(candidate):
            
            state[f'reducto_{keys[idx]}_bias'] = torch.tensor(val * 1.0)
        
        with torch.no_grad():
            video, metrics = reducto_process(gt_video,state)
        
        prob_matrix = metrics['prob_matrix']
        # calculate objective
        average_accuracy = 0
        for cur_id in range(len(gt_video)):
            for prev_id in range(len(gt_video)):
                if prob_matrix[cur_id, prev_id] > 0:
                    # attach a frame id 0 to the inference results, to fit the format of calc_accuracy
                    average_accuracy = average_accuracy + prob_matrix[cur_id, prev_id] * app.calc_accuracy({0:results[prev_id]}, {0:results[cur_id]})['f1']
                
        reconstruction = -average_accuracy
        compute = settings.backprop.compute_weight * metrics['compute']
        
        if min_objective == {} or reconstruction + compute < min_objective['rec'] + min_objective['com']:
            min_objective['rec'] = reconstruction
            min_objective['com'] = compute
            min_state = deepcopy(state)
                
    # recover the min state
    for key in state:
        state[key] = min_state[key]
    
    return min_objective


def grad_reducto_cheap(state, gt_video, saliency):
    """Optimize reducto thresholds cheaply through grid search

    Args:
        state (dict): the state dict that includes reducto thresholds
        gt_video (torch.tensor): the ground-truth video
        saliency (torch.tensor): the saliency on the video
    
    Returns:
        dict: the optimal objective function
    """  
    
    keys = list(reducto_feature2meanstd.keys())
    
    min_objective = {}
    min_state = {}
    
    for candidate in tqdm(list(product(*[[-2, 1,0, 1, 2] for key in keys])), desc='reducto cheap search'):

        for idx, val in enumerate(candidate):
            
            state[f'reducto_{keys[idx]}_bias'] = torch.tensor(val * 1.0)

        with torch.no_grad():
            video, metrics = reducto_process(gt_video,state)
        
        # calculate objective
        reconstruction = (saliency * (gt_video - video).abs()).mean()
        compute = settings.backprop.compute_weight * metrics['compute']
        logger.info('state: %s, rec: %.3f, com: %.3f', candidate, reconstruction, compute)
        
        if min_objective == {} or reconstruction + compute < min_objective['rec'] + min_objective['com']:
            min_objective['rec'] = reconstruction
            min_objective['com'] = compute
            min_state = deepcopy(state)
            
    # recover the min state
    for key in state:
        state[key] = min_state[key]

    vis.text += "rec: %.3e, com: %.3e\n" % (min_objective['rec'], min_objective['com'])
        
    return min_objective

        

def grad(args: dict, key: str, grad: torch.Tensor, gt_config, gt_video):
 
    if key == 'macroblocks':
        return optimize_macroblocks(args, key, grad, gt_config, gt_video)
    
    # assert not hq_video.requires_grad

    

    configs = conf.space[key]
    args = args.copy()

    # set_trace()

    hq_index = configs.index(args[key])
    lq_index = hq_index + 1
    assert lq_index < len(configs)
    delta = 1.0 / (len(configs) - 1)
    x = state[key]
    if x.grad is None:
        x.grad = torch.zeros_like(x)

    def check():

        # logger.info(f'Index: HQ {hq_index} and LQ {lq_index}')

        logger.info(f'Searching {key} between HQ {configs[hq_index]} and LQ {configs[lq_index]}')
        
        args[key] = configs[lq_index]
        lq_name, lq_config = encode(args)
        lq_video = read_video_to_tensor(lq_name)
        # print(lq_config)

        args[key] = configs[hq_index]
        hq_name, hq_config = encode(args)
        hq_video = read_video_to_tensor(hq_name)
        # print(hq_config)

        diff_thresh = settings.backprop.difference_threshold

        hq_diff = (hq_video - gt_video).abs()
        hq_diff = hq_diff.sum(dim=1, keepdim=True)
        # logger.info(f'%.5f pixels in HQ are neglected', (hq_diff>diff_thresh).sum()/(hq_diff>-1).sum()) 
        # hq_diff[hq_diff > diff_thresh] = 0
        
        lq_diff = (lq_video - gt_video).abs()
        lq_diff = lq_diff.sum(dim=1, keepdim=True)
        # logger.info(f'%.5f pixels in LQ are neglected', (lq_diff>diff_thresh).sum()/(lq_diff>-1).sum()) 
        # lq_diff[lq_diff > diff_thresh] = 0

        

        if (hq_video - lq_video).abs().mean() > 1e-5: 

            logger.info('Search completed.')
            
            left, right = 1 - delta * hq_index, 1 - delta * lq_index
            # print(f'Left {left}, x {x}, right {right}')
            if not left >= x > right:
                breakpoint()
            assert left >= x > right


            # accuracy gradient term
            # wanna minimize this. So positive gradient.
            lq_ratio = (left - x) / (left - right)
            hq_ratio = (x - right) / (left - right)

            hq_losses = {
                'rec': (grad * hq_diff).abs().mean().item(),
                'bw': hq_config['bw'] / gt_config['bw'],
                'com': len(hq_config['encoded_frames']) / settings.segment_length
            }

            lq_losses = {
                'rec': (grad * lq_diff).abs().mean().item(),
                'bw': lq_config['bw'] / gt_config['bw'],
                'com': len(lq_config['encoded_frames']) / settings.segment_length
            }

            delta_loss = {i: hq_losses[i] - lq_losses[i] for i in hq_losses.keys()}

            logger.info(f'{key}({configs[hq_index]}): {hq_losses}')
            logger.info(f'{key}({configs[lq_index]}): {lq_losses}')
            logger.info(f'Delta: {delta_loss}')
            
            logger.info('HQ loss: %.4f', sum(hq_losses.values()))
            logger.info('LQ loss: %.4f', sum(lq_losses.values()))

            objective = {i: hq_losses[i] * hq_ratio + lq_losses[i] * lq_ratio for i in hq_losses.keys()}
            objective = settings.backprop.reconstruction_loss_weight * objective['rec'] + settings.backprop.bw_weight * objective['bw'] + settings.backprop.compute_weight* objective['com']
            
            
            logger.info(f'Before backwrad, gradient is {x.grad}')

            objective.backward()

            logger.info('Gradient is %.3f', x.grad)
            

            return True

        else:

            return False   


    while (hq_index > 0 or lq_index < len(configs) - 1):

        if check():
            return
        else:
            # directly return if the gradient is zero. Just to speed up the runtime.
            return
        
        hq_index -= 1
        hq_index = max(hq_index, 0)
        lq_index += 1
        lq_index = min(lq_index, len(configs) - 1)
    
    check()

