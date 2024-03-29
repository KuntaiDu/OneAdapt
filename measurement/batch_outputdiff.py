
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
    f'/dataheart/dataset/downtown/downtown_{i}/part%d.mp4' for i in range(10)
]

st, ed = 0, 10

# for qp, fr, res, bwweight in product(qp_list, fr_list, res_list, bwweight_list):
for idx, fmt in enumerate(fmts):
    if idx % 2 == 1:
        continue
    env = os.environ.copy()
    env['SETTINGS_FILE'] = '/dataheart/kuntai_recovery/code/diff_yitian/measurement/settings_outputdiff.toml'
    run([
        'python', 'outputdiff.py',
        '-i', fmt,
        # '-i', 'videos/yoda/dashcam_1/part%d.mp4',
        # '--sec', '61',
        '--start', '%d' % st,
        '--end', '%d' % ed,
        '--stats', '/dataheart/kuntai_recovery/code/diff_yitian/measurement/stats/temp7.yaml'
        # '--qp', f'{qp}',
        # '--res', f'{res}',
        # '--fr', f'{fr}',
        # '--lr', f'{lr}',
        # '--freq', f'{freq}',
        # '--train',
        # '--bw_weight', f'{bwweight}',
    ], env=env)
