
import torch
from munch import *
from pdb import set_trace

__all__ = ['get_frame_features', 'get_features']






def mean_sum_score(feats, args):

    return {'mean_sum_score': feats[:, 0].mean().item()}


def std_sum_score(feats, args):

    return {'std_sum_score': feats[:, 0].std(unbiased=False).item()}


def mean_count_score(feats, args):

    return {'mean_count_score': feats[:, 1].mean().item()} 


def std_count_score(feats, args):

    return {'std_count_score': feats[:, 1].std(unbiased=False).item()}


feature_list = [
    mean_sum_score,
    std_sum_score,
    mean_count_score, 
    std_count_score
]

def get_frame_features(scores, args):
    


    feat = torch.cat(
        [
            scores.sum()[None], 
            ((scores - args.confidence_threshold) * 10).sigmoid().sum()[None],
            (scores > args.confidence_threshold).float().sum()[None],
            ],
        )
    
    return feat.unsqueeze(0)


def get_features(feats, args):

    ret = {}
    for func in feature_list:
        feat = func(feats, args)
        ret.update(feat)

    return ret