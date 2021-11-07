
import subprocess

for fr in [1,3,5,15]:

    subprocess.run([
        'python',
        'awstream.py',
        '--hq',
        './vis_171_qp_24.mp4',
        '--lq',
        './vis_171_qp_44.mp4',
        '--fr',
        f'{fr}'
    ])