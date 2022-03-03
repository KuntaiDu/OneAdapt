
from itertools import product
from subprocess import run
import os
from pathlib import Path


def probe_range(fmt):
    
    idx = 0
    while Path(fmt % idx).exists():
        idx += 1
    return idx

# fmts = [
#     f'videos/yoda/dashcam_{i}/part%d.mp4' for i in range(1, 9)
# ]
fmts = [
    f'videos/driving/driving_{i}/part%d.mp4' for i in range(5)
]

# for qp, fr, res, bwweight in product(qp_list, fr_list, res_list, bwweight_list):
for fmt in fmts:
    
    for bw_weight in [0.8, 0.4, 0.2, 0.1, 0.05]:
        
        for downsample_factor in [1]:

            # output = f'diff_results_dense_interp/stuttgart_0_lr_{lr}_qp_{qp}_res_{res}_fr_{fr}.txt'
            # output = f'stats/diff_results_reducto/reducto-efficientdet-d2.txt'
            # approach = 'backprop_30_threshold_loss_iterative_training_new_lr_7e-4_cheat_saliency_error'

            # loss_type = 'saliency_error'

            approach = f'chameleon_top3_bwweight_{bw_weight}'
            
            env = os.environ.copy()
            
            env['DYNACONF_BACKPROP__BW_WEIGHT'] = f'{bw_weight}'
            

            run([
                'python', 'chameleon.py',
                '-i', fmt,
                # '-i', 'videos/yoda/dashcam_1/part%d.mp4',
                # '--sec', '61',
                '--start', '0',
                '--end', '%d' % probe_range(fmt),
                '--frequency', '100',
                # '--qp', f'{qp}',
                # '--res', f'{res}',
                # '--fr', f'{fr}',
                # '--lr', f'{lr}',
                # '--freq', f'{freq}',
                # '--train',
                '--approach', approach,
                '--downsample_factor', f'{downsample_factor}',
                '--enable_top3',
                # '--bw_weight', f'{bwweight}',
            ], env=env)

            # freq = 1

            # output = f'diff_results_dense_interp/stuttgart_0_diff_freq_{freq}_lr_{lr}_qp_{qp}_res_{res}_fr_{fr}.txt'

            # if not os.path.exists(output):

            #     run([
            #         'python', 'diff.py',
            #         '-i', 'cityscape/stuttgart_0/%d/video',
            #         '--sec', '20',
            #         '--qp', f'{qp}',
            #         '--res', f'{res}',
            #         '--fr', f'{fr}',
            #         '--lr', f'{lr / orig_freq}',
            #         '--freq', f'{freq}',
            #         '--train',
            #         '--output', output
            #     ])
