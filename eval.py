"""
    Compress the video through gradient-based optimization.
"""

import pandas as pd
import yaml
import numpy as np
from pdb import set_trace


def get_config(i):

    return (i['fr'], i['qp'], i['res'])

def get_performance(stats, start_sec, end_sec):

    norm_bws = []
    f1s = []
    
    for t in range(start_sec, end_sec):
        
        result = [i for i in stats if ('/%d/' % t) in i['video_name']]
        result = result[-1]

        norm_bws.append(result['norm_bw'])
        f1s.append(result['f1'])

    return (np.array(norm_bws).mean().item(), np.array(f1s).mean().item())




def get_awstream(stats, start_sec, end_sec):


    profile = [i for i in stats if '/0/' in i['video_name']]

    # profile = stats[stats['video_name'].str.contains('/0/')]


    # obtain pareto boundary
    for i in profile:
        i['discard'] = False
    for i in profile:
        for j in profile:
            if i['bw'] < j['bw'] and i['f1'] > j['f1']:
                j['discard'] = True
    profile = [i for i in profile if not i['discard']]
    
    
    # obtain config bundle
    configs = [get_config(i) for i in profile]


    performances = []

    # obtain the config for each item in the config bundle
    for config in configs:

        filtered_stats = [i for i in stats if get_config(i) == config]
        performances.append(get_performance(filtered_stats, start_sec, end_sec))

    return performances


def get_all(stats, start_sec, end_sec):


    profile = [i for i in stats if '/0/' in i['video_name']]

    # profile = stats[stats['video_name'].str.contains('/0/')]


    # obtain pareto boundary
    
    
    # obtain config bundle
    configs = [get_config(i) for i in profile]


    performances = []

    # obtain the config for each item in the config bundle
    for config in configs:

        filtered_stats = [i for i in stats if get_config(i) == config]
        performances.append(get_performance(filtered_stats, start_sec, end_sec))

    return performances


def get_diff_performance(stats, state, t):

    stats = [i for i in stats if (state[0] % t) == i['video_name']]

    return state[1] * stats[-1]['norm_bw'], state[1] * stats[-1]['f1']




def get_diff(stats, diff_file, freeze_sec, start_sec, end_sec):

    diffs = yaml.load(open(diff_file, 'r'))

    norm_bws = []
    f1s = []
    all_states = None

    for diff in diffs:
        if diff['sec'] == freeze_sec:
            all_states = diff['all_states']
            break

        norm_bws.append(diff['true_average_bw'])
        f1s.append(diff['true_average_f1'])



    for sec in range(freeze_sec, end_sec):

        norm_bws_sec = []
        f1s_sec = []

        for state in all_states:
            norm_bw, f1 = get_diff_performance(stats, state, sec)
            norm_bws_sec.append(norm_bw)
            f1s_sec.append(f1)

        norm_bws.append(np.array(norm_bws_sec).sum().item())
        f1s.append(np.array(f1s_sec).sum().item())

    assert len(norm_bws) == end_sec


    norm_bws = norm_bws[start_sec:end_sec]
    f1s = f1s[start_sec:end_sec]

    return (np.array(norm_bws).mean().item(), np.array(f1s).mean().item())
