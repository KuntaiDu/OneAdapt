
from subprocess import run
import os
from pathlib import Path
import glob

video_list = [
    ('/tank/kuntai/visdrone/VisDrone2019-SOT-train/sequences/uav0000003_00000_s/', 3),
]

for v, idx in video_list:
    
    length = len(glob.glob(v + '/*.jpg'))
    path = Path('videos/dashcam/dashcam_%d/' % idx)
    os.system(f'rm -r {path}')
    path.mkdir(parents=True)
    
    for time in range((length//10) - 1):
        
        start = time * 10
        
        run([
            'ffmpeg', '-framerate', '10', '-start_number', f'{start}', '-i', v + '/img%07d.jpg',  '-frames:v', '10', '-c:v', 'libx264', '-qp', '0', 'videos/dashcam/dashcam_%d/part%d.mp4' % (idx, time)
        ])

    