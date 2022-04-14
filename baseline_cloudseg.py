"""
    Compress the video through gradient-based optimization.
"""

import argparse
import gc
import logging
import time
from pathlib import Path
from typing import Tuple

import coloredlogs
import enlighten
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import pickle
import torchvision.transforms as T
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from utils.visualize_utils import visualize_heat_by_summarywriter
from torchvision import io
from datetime import datetime
import random
import torchvision

import yaml
from config import settings

from pdb import set_trace

from dnn.dnn_factory import DNN_Factory
from dnn.dnn import DNN
# from utils.results import write_results
from utils.video_reader import read_video, read_video_config
import utils.config_utils as conf
from collections import defaultdict
from tqdm import tqdm
from inference import inference, encode
from examine import examine
import pymongo
from munch import *


sns.set()


# set_trace()
conf.space = munchify(settings.configuration_space.to_dict())
state = {}

len_gt_video = 10
logger = logging.getLogger("diff")



# default_size = (800, 1333)
conf.serialize_order = list(settings.backprop.tunable_config.keys())



def augment(result, lengt):

    factor = (lengt + (len(result) - 1)) // len(result)

    return torch.cat([result[i // factor][None, :, :, :] for i in range(lengt)])




def read_expensive_from_config(gt_args: Munch, state, app: DNN, db: pymongo.database.Database) -> Tuple[dict, Munch]:

    average_video = None
    average_bw = 0
    sum_prob = 0

    # ret = defaultdict(lambda: 0)
    
    for args in conf.serialize_most_expensive_state(gt_args.copy(), conf.state2config(state), conf.serialize_order):


        # encode
        # args['gamma'] = 1.0
        video_name, remaining_frames = encode(args)
        video = list(read_video(video_name))
        video = torch.cat([i[1] for i in video])
        # video = F.interpolate(video, size=default_size)
        video = augment(video, len_gt_video)

        
        # video = video * prob

        # sum_prob += prob

        # if average_video is None:
        #     average_video = video
        # else:
        #     average_video = average_video + video

        # update statistics of random choice.
        stat = examine(args,gt_args,app,db)

        return stat, args, video


        # assert ret.keys() == {}.keys() or stat.keys() == ret.keys()

    #     for key in stat:
    #         if type(stat[key]) in [int, float]:
    #             ret[key] += stat[key] * prob

    #     Path(video_name).unlink()


    # ret.update({'video': average_video})
    # ret = dict(ret)
    # return ret


def optimize(args: dict, key: str, grad: torch.Tensor):

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
        lq_name, lq_remaining_frames = encode(args)
        lq_video = torch.cat([i[1] for i in list(read_video(lq_name))])
        print(lq_remaining_frames)

        args[key] = configs[hq_index]
        hq_name, hq_remaining_frames = encode(args)
        hq_video = torch.cat([i[1] for i in list(read_video(hq_name))])
        print(hq_remaining_frames)

        

        if (hq_video - lq_video).abs().mean() > 1e-5: 

            logger.info('Search completed.')
            
            left, right = 1 - delta * hq_index, 1 - delta * lq_index
            assert left >= x > right

            

            x.grad += ( ((hq_video - lq_video) / (left - right)) * grad ).sum()

            return True

        else:

            return False   


    while (hq_index > 0 or lq_index < len(configs) - 1):

        if check():
            return
        
        hq_index -= 1
        hq_index = max(hq_index, 0)
        lq_index += 1
        lq_index = min(lq_index, len(configs) - 1)
    
    check()








def main(command_line_args):


    # a bunch of initialization.
    
    torch.set_default_tensor_type(torch.FloatTensor)

    db = pymongo.MongoClient("mongodb://localhost:27017/")[settings.collection_name]

    app = DNN_Factory().get_model(settings.backprop.app)

    writer = SummaryWriter(f"runs/{command_line_args.approach}")

    logger.info("Application: %s", app.name)
    logger.info("Input: %s", command_line_args.input)
    logger.info("Approach: %s", command_line_args.approach)
    progress_bar = enlighten.get_manager().counter(
        total=command_line_args.end - command_line_args.start,
        desc=f"{command_line_args.input}",
        unit="10frames",
    )

        
    # initialize configurations pace.
    parameters = []
    for key in settings.backprop.tunable_config.keys():
        if key == 'cloudseg':
            parameters.append({
                "params": conf.SR_dnn.net.parameters(), 
                "lr": settings.backprop.tunable_config_lr[key]
            })
            continue
        state[key] = torch.tensor(settings.backprop.tunable_config[key])
        parameters.append({
            "params": state[key],
            "lr": settings.backprop.tunable_config_lr[key]
        })


    # build optimizer
    for tensor in state.values():
        tensor.requires_grad = True
    if settings.backprop.tunable_config.cloudseg:
        for param in conf.SR_dnn.net.parameters():
            param.requires_grad = True
    optimizer = torch.optim.Adam(parameters, lr=settings.backprop.lr)



    for sec in range(command_line_args.start, command_line_args.end):

        progress_bar.update()

        logger.info('\nAt sec %d', sec)

        
        gt_args = munchify(settings.ground_truths_config.to_dict()) 
        gt_args.update({
            'input': command_line_args.input,
            'second': sec // 10,
        })



        # construct average video and average bw
        ret, args, video = read_expensive_from_config(gt_args, state, app, db)
        gt_video_name, _ = encode(gt_args)
        gt_video = list(read_video(gt_video_name))
        gt_video = torch.cat([i[1] for i in gt_video])
        
        
        if 'gamma' in state:
            video = (video ** state['gamma']).clamp(0, 1)

        args.cloudseg = True
        args.approach = command_line_args.approach
        
        # results = pickle.loads(inference(args, db, app)['inference_result'])
        gt_results = pickle.loads(inference(gt_args, db, app)['inference_result'])
        # for idx, frame in enumerate(tqdm(video)):
        #     with torch.no_grad():
        #         if settings.backprop.tunable_config.cloudseg:
        #             frame = conf.SR_dnn(frame.unsqueeze(0).to('cuda:1')).cpu()
        #         result = app.inference(frame, detach=True, grad=False)
        #         score = torch.sum(result["instances"].scores)
        #         scores[idx] = score
        # scores = {}
        # for idx in results:
        #     scores[key] = torch.sum(results[idx]['instances'].scores)

        # interpolated_fr = conf.state2config(state)['fr']
        # interpolated_fr = interpolated_fr[0][0] * interpolated_fr[0][1] + interpolated_fr[1][0] * interpolated_fr[1][1]

        # set_trace()
        
        # interpolated_fr = ret['#remaining_frames']
        # average_std_score_mean = torch.tensor([scores[i] for i in scores.keys()]).var(unbiased=False).detach()
        # average_sum_score = torch.tensor([scores[i] for i in scores.keys()]).mean()
        # sum_score = torch.tensor([scores[i] for i in scores.keys()]).detach().cpu()



        if settings.backprop.train:

            # take the gradient from the video
            # video.requires_grad = True

            saliencies = {}

            # calculate saliency on each frame.
            for idx, frame in enumerate(tqdm(video)):
                gt_frame = gt_video[idx]
                if settings.backprop.tunable_config.cloudseg:
                    with torch.no_grad():
                        frame = conf.SR_dnn(frame.unsqueeze(0).to('cuda:1')).cpu()

                frame_detached = frame.detach()
                frame_detached.requires_grad_()
                result = app.inference(frame_detached, detach=False, grad=True)
                # filter out unrelated classes
                result = app.filter_result(result, confidence_check=False)
                score = result["instances"].scores
                score_inds = (settings[app.name].confidence_threshold < score) & (score < settings[app.name].confidence_threshold + 0.1)
                score = score[score_inds]
                sum_score = torch.sum(score)
                
                sum_score.backward()

                saliency = frame_detached.grad.abs().sum(dim=1, keepdim=True)
                
                # average across 16x16 neighbors
                kernel = torch.ones([1, 1, command_line_args.average_window_size, command_line_args.average_window_size])
                saliency = F.conv2d(saliency, kernel, stride=1, padding=(command_line_args.average_window_size - 1) // 2)
                saliencies[idx] = saliency


            for iteration in range(command_line_args.num_iterations):

                for idx, frame in enumerate(tqdm(video)):

                    if settings.backprop.tunable_config.cloudseg:
                        frame = conf.SR_dnn(frame.unsqueeze(0).to('cuda:1')).cpu()

                    # # for visualization purpose.
                    # result = app.inference(frame.detach(), detach=False, grad=False)

                    # calculate saliency again
                    frame_detached = frame.detach()
                    frame_detached.requires_grad_()
                    result = app.inference(frame_detached, detach=False, grad=True)
                    # filter out unrelated classes
                    result = app.filter_result(result, confidence_check=False)
                    score = result["instances"].scores
                    score_inds = (settings[app.name].confidence_threshold < score) & (score < settings[app.name].confidence_threshold + 0.1)
                    score = score[score_inds]
                    sum_score = torch.sum(score)
                    
                    sum_score.backward()

                    saliency = frame_detached.grad.abs().sum(dim=1, keepdim=True)
                    kernel = torch.ones([1, 1, command_line_args.average_window_size, command_line_args.average_window_size])
                    saliency = F.conv2d(saliency, kernel, stride=1, padding=(command_line_args.average_window_size - 1) // 2)

                    # reconstruction_loss = (saliencies[idx] * (gt_video[idx].unsqueeze(0) - frame).abs()).mean()
                    reconstruction_loss = (saliency * (gt_video[idx].unsqueeze(0) - frame).abs()).mean()
                    (saliencies[idx] * (gt_video[idx].unsqueeze(0) - frame).abs()).mean()
                    reconstruction_loss.backward()

                    writer.add_scalar('Reconstruction/%d' % idx, reconstruction_loss.item(), iteration)

                    if settings.backprop.early_optimize:
                        logger.info('Early optimize')
                        optimizer.step()
                        optimizer.zero_grad()


                    if idx % 3 == 0:
                        with torch.no_grad():
                            for key in result['instances'].get_fields():
                                if key == 'pred_boxes':
                                    result['instances'].get(key).tensor = result['instances'].get(key).tensor.detach().cpu()
                                else:
                                    result['instances'].set(key, result['instances'].get(key).detach().cpu())
                            gt_ind, res_ind, gt_filtered, res_filtered = app.get_undetected_ground_truth_index(result, gt_results[idx])
                            image = T.ToPILImage()(frame.detach()[0])
                            image_FN = app.visualize(image, {'instances': gt_filtered[gt_ind]})
                            image_FP = app.visualize(image, {'instances': res_filtered[res_ind]})
                            visualize_heat_by_summarywriter(image_FN, saliency[0][0], f'FN/{idx}', writer, iteration, tile=False)
                            visualize_heat_by_summarywriter(image_FP, saliency[0][0], f'FP/{idx}', writer, iteration, tile=False)

                        if iteration == 0:
                            writer.add_image('GT/%d' % idx, gt_video[idx], idx)


                for key in settings.backprop.tunable_config.keys():
                    if key == 'cloudseg':
                        # already optimized when backwarding.
                        continue
                    optimize(args, key, video.grad)

        # if args.train:
        #     (-(1/len_gt_video) * last_score).backward(retain_graph=True)
        # average_sum_score = average_sum_score +  (-(1/len_gt_video) * last_score).item()
            



        # objective = (settings.backprop.sum_score_mean_weight * average_sum_score + settings.backprop.std_score_mean_weight * average_std_score_mean)
        # true_obj = (settings.backprop.sum_score_mean_weight * true_average_score + settings.backprop.std_score_mean_weight * true_average_std_score_mean  - settings.backprop.compute_weight * interpolated_fr.detach().item())
        
        
        state_str = ""
        for key in conf.serialize_order:
            if key == 'cloudseg':
                param = list(conf.SR_dnn.net.parameters())[0]
                logger.info('CloudSeg: gradient of layer 0 mean: %.3f, std: %.3f', param.grad.mean(), param.grad.std())
                continue
            logger.info('%s : %.3f, grad: %.7f', key, state[key], state[key].grad)
            state_str += '%s : %.3f, grad: %.7f\n' % (key, state[key], state[key].grad)

        # logger.info('QP: %.3f, Res: %.3f, Fr: %.3f', state['qp'], state['res'], state['fr'])
        # logger.info('qpgrad: %.3f, frgrad: %.3f, resgrad: %.3f', state['qp'].grad, state['fr'].grad, state['res'].grad)
        
        # logger.info('Score: %.3f, std: %.3f, bw : %.3f, Obj: %.3f', average_sum_score, average_std_score_mean, ret['bw'], objective.item())
        logger.info('Reconstruction loss: %.3f', reconstruction_loss.item())

        performance = examine(args, gt_args, app, db)
        logger.info('F1: %.3f', performance['f1'])

        # logger.info('True : %.3f, Tru: %.3f, Tru: %.3f, Tru: %.3f', true_average_score, true_average_std_score_mean, true_average_bw, true_obj)

        # optimize

        # truncate
        if (not settings.backprop.early_optimize) and settings.backprop.train and (sec + 1) % settings.backprop.freq == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        for tensor in state.values():
            tensor.requires_grad = False
        for key in conf.serialize_order:
            if key == 'cloudseg':
                continue
            if state[key] > 1. :
                state[key][()] = 1.
            if state[key] < 1e-7:
                state[key][()] = 1e-7
        for tensor in state.values():
            tensor.requires_grad = True
        

        
        logger.info(f'Current config: {conf.state2config(state)}')

        logger.info(f'Current state: {state}')

        # choose = conf.random_serialize(video_name, conf.state2config(state))

        
        # # logger.info('Choosing %s', choose)

        # with open(args.output, 'a') as f:
        #     f.write(yaml.dump([{
        #         'sec': sec,
        #         # 'choice': choose,
        #         'config': conf.state2config(state, serialize=True),
        #         'true_average_bw': true_average_bw,
        #         'true_average_score': true_average_score,
        #         'true_average_f1': true_average_f1,
        #         'fuse_obj': fuse_obj.item(),
        #         'true_obj': true_obj,
        #         'average_sum_score': average_sum_score.item(),
        #         'average_std_score_mean': average_std_score_mean.item(),
        #         'average_range_score_mean': ((sum_score.max() - sum_score.min()) / interpolated_fr).item(),
        #         'average_abs_score_mean': (sum_score - sum_score.mean()).abs().mean().item(),
        #         'state': state_str,
        #         # 'all_states': list(conf.serialize_all_states(args.input, conf.state2config(state, serialize=True), 1., conf.serialize_order)),
        #         # 'qp_grad': state['qp'].grad.item()
        #     }]))

        

        

        
            

        # set_trace()

    # for idx, (hqs, lqs) in enumerate(zip(read_video(args.hq, args), read_video(args.lq, args))):

    #     hqs = torch.cat([i[1] for i in hqs])
    #     lqs = torch.cat([i[1] for i in lqs])

    #     # frames = fr(q(hqs, lqs))
    #     frames = q(hqs, lqs)
    #     # frames = fr(hqs)


    #     for frame, hq in zip(frames, hqs):

    #         progress_bar.update()

    #         with torch.no_grad():
    #             result = app.inference(frame.unsqueeze(0), detach=True)
    #         # with torch.no_grad():
    #         #     hq_result = app.inference(hq.unsqueeze(0), detach=True)
    #         inference_results[fid] = result
            
    #         if idx % args.freq == 0:
    #             activation = app.activation(frame.unsqueeze(0))
    #             activation.backward(retain_graph=True)

    #         fid += 1

    #     if idx % args.freq == 0:
    #         # fr.step()
    #         q.step()

    #         image = F.interpolate(hqs, size=(480, 640))
    #         image = T.ToPILImage()(image[0])
    #         image = app.visualize(image, result, args)
    #         writer.add_image('inference', T.ToTensor()(image), fid)
            
    #         q.visualize(hqs[0], fid)

    #         means.append(q.q.detach().mean())


    mean = torch.tensor(means).mean().item()

    logger.info('Overall mean quality: %.3f', mean)

    # with open('config.yaml', 'a') as f:
    #     f.write(yaml.dump([{
    #         '#frames': fid,
    #         'bw': mean * Path(args.hq).stat().st_size + (1-mean) * Path(args.lq).stat().st_size,
    #         'video_name': args.output
    #     }]))

    # with open('diff.yaml', 'a') as f:
    #     f.write(yaml.dump({
    #         'acc': accs,
    #         'compute': computes,
    #         'size': sizes
    #     }))

    # print(torch.tensor(accs).mean() + torch.tensor(computes).mean() + torch.tensor(sizes).mean())
        


if __name__ == "__main__":

    # set the format of the logger
    coloredlogs.install(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s",
        level="INFO",
    )

    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--freq",
    #     help="The video file names. The largest video file will be the ground truth.",
    #     default=1,
    #     type=int
    # )

    # parser.add_argument(
    #     '--lr',
    #     help='The learning rate',
    #     type=float,
    #     default=0.003
    # )

    # parser.add_argument(
    #     '--qp',
    #     help='The quantization parameter',
    #     type=float,
    #     default=1.
    # )

    # parser.add_argument(
    #     '--fr',
    #     help='The frame rate',
    #     type=float,
    #     default=1.
    # )

    # parser.add_argument(
    #     '--res',
    #     help='The resolution',
    #     type=float,
    #     default=1.
    # )

    parser.add_argument(
        '-i',
        '--input',
        help='The format of input video.',
        type=str,
        required=True
    )

    parser.add_argument(
        '--start',
        help='The total secs of the video.',
        required=True,
        type=int
    )

    parser.add_argument(
        '--end',
        help='The total secs of the video.',
        required=True,
        type=int
    )

    parser.add_argument(
        '--num_iterations',
        help='The total secs of the video.',
        required=True,
        type=int
    )


    parser.add_argument(
        "--app", 
        type=str, 
        help="The name of the model.", 
        default='EfficientDet-d2',
    )

    parser.add_argument(
        "--average_window_size",
        type=int,
        help='The window size for saliency averaging',
        default=17
    )

    parser.add_argument(
        '--approach',
        type=str,
        required=True
    )
    # parser.add_argument(
    #     '--gamma',
    #     type=float,
    #     help='Adjust the luminance.',
    #     default=1.5,
    # )

    args = parser.parse_args()

    main(args)
