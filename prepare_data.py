
from subprocess import run
import os
from pathlib import Path
import glob

video_list = [
    ('/tank/kuntai/code/video-compression/videos/driving_0', 0),
    ('/tank/kuntai/code/video-compression/videos/driving_1', 1),
    ('/tank/kuntai/code/video-compression/videos/driving_2', 2),
    ('/tank/kuntai/code/video-compression/videos/driving_3', 3),
    ('/tank/kuntai/code/video-compression/videos/driving_4', 4),
]

for v, idx in video_list:
    
    length = len(glob.glob(v + '/*.png'))
    path = Path('videos/driving/driving_%d' % idx)
    os.system(f'rm -r {path}')
    path.mkdir(parents=True)
    
    for time in range((length//10) - 1):
        
        start = time * 10
        
        run([
            'ffmpeg', '-framerate', '10', '-start_number', f'{start}', '-i', v + '/%010d.png',  '-frames:v', '10', '-c:v', 'libx264', '-qp', '0', 'videos/driving/driving_%d/part%d.mp4' % (idx, time)
        ])

    