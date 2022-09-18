
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

lr = 0.1

# lr = 0
orig_freq = 1

force = True


def probe_range(fmt):
    
    idx = 0
    while Path(fmt % idx).exists():
        idx += 1
    return idx

# fmts = [
#     f'videos/trafficcam/trafficcam_{i}/part%d.mp4' for i in range(1, 2)
# ]
# fmts = [
#     f'videos/trafficcam/trafficcam_{i}/part%d.mp4' for i in range(1, 2)
# ]

# fmts = [
#     f'videos/dashcamcropped/dashcamcropped_1_3xdownsample/part%d.mp4'
# ]

# 1fps Accuracy > 60%

fmts = [
    f'/dataheart/dataset/country/country_{i}/part%d.mp4' for i in [1,3]
]
st, ed = 0, 119

# for qp, fr, res, bwweight in product(qp_list, fr_list, res_list, bwweight_list):

# varying profile frequency does not affect performance much.
for idx, fmt in enumerate(fmts):

    # if idx != 2:
    #     continue
    for compute_weight in [0.01, 0.04, 0.07]:
        for freq in [4]:

            if fmt != "/dataheart/dataset/country/country_1/part%d.mp4":
                continue


            # st, ed = 41, 51
            # st, ed = 120, 130

            # output = f'diff_results_dense_interp/stuttgart_0_lr_{lr}_qp_{qp}_res_{res}_fr_{fr}.txt'
            # output = f'stats/diff_results_reducto/reducto-efficientdet-d2.txt'
            # approach = 'backprop_30_threshold_loss_iterative_training_new_lr_7e-4_cheat_saliency_error'

            loss_type = 'None'

            approach = approach = f'reducto_weight_{compute_weight}_freq_{freq}'
            

            if force or not os.path.exists(output):
                
                env = os.environ.copy()
                
                env['DYNACONF_BACKPROP__COMPUTE_WEIGHT'] = f'{compute_weight}'
                env['SETTINGS_FILE'] = 'settings_reducto.toml'
                env['DYNACONF_BACKPROP__REDUCTO_EXPENSIVE_OPTIMIZE'] = 'true'

                run([
                    'python', 'diff_reducto.py',
                    '-i', fmt,
                    # '-i', 'videos/yoda/dashcam_1/part%d.mp4',
                    # '--sec', '61',
                    '--start', f'{st}',
                    # '--end', '%d' % probe_range(fmt),
                    '--end',  f'{ed}',
                    '--num_iterations', '1', 
                    '--loss_type', loss_type,
                    '--frequency', f'{freq}',
                    # '--qp', f'{qp}',
                    # '--res', f'{res}',
                    # '--fr', f'{fr}',
                    # '--lr', f'{lr}',
                    # '--freq', f'{freq}',
                    # '--train',
                    '--approach', approach,
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
