
from itertools import product
from subprocess import run
from pathlib import Path
import os

# qp_list = [1e-6,  0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
# qp_list = [0.95]
# qp_list = qp_list[::-1]

# qp_list = [qp_list[i] for i in range(len(qp_list)) if i % 2 == 0]
# qp_list = [1]
# qp_list = [1]
# res_list = [1]
# fr_list = [1e-6,  0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
# fr_list = [fr_list[i] for i in range(len(fr_list)) if i % 2 == 1]
# res_list = [1e-6, 0.49, 1]
# fr_list = res_list
# qp_list = res_list

res_list = [1]
fr_list = [1]
qp_list = [1]
bwweight_list = [6]
# res_list = [1e-6,  0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
# res_list = [res_list[i] for i in range(len(res_list)) if i % 2 == 1]

lr = 0.2

# lr = 0
orig_freq = 1

force = True


def probe_range(fmt):
    
    idx = 0
    while Path(fmt % idx).exists():
        idx += 1
    return idx

fmts = [
    f'videos/yoda/dashcam_{i}/part%d.mp4' for i in range(1, 9)
]

# for qp, fr, res, bwweight in product(qp_list, fr_list, res_list, bwweight_list):
for fmt in fmts:

    freq = orig_freq

    # output = f'diff_results_dense_interp/stuttgart_0_lr_{lr}_qp_{qp}_res_{res}_fr_{fr}.txt'
    # output = f'stats/diff_results_reducto/reducto-efficientdet-d2.txt'
    # approach = 'backprop_30_threshold_loss_iterative_training_new_lr_7e-4_cheat_saliency_error'

    loss_type = 'saliency_error'

    approach = f'backprop_sigmoid_{loss_type}_add_preset_lr_0.1_9sec'
    

    if force or not os.path.exists(output):

        run([
            'srun', '-p', 'general', '--gres=gpu:1', '-w', 'aa[001-003]', '--pty',
            'python', 'diff_cloudseg.py',
            '-i', fmt,
            # '-i', 'videos/yoda/dashcam_1/part%d.mp4',
            # '--sec', '61',
            '--start', '0',
            '--end', '%d' % probe_range(fmt),
            '--num_iterations', '1',
            '--loss_type', loss_type,
            '--frequency', '3',
            # '--qp', f'{qp}',
            # '--res', f'{res}',
            # '--fr', f'{fr}',
            # '--lr', f'{lr}',
            # '--freq', f'{freq}',
            # '--train',
            '--approach', approach,
            # '--bw_weight', f'{bwweight}',
        ])

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
