
from subprocess import run
import os
from pathlib import Path
import glob

video_list = [
    # ('/tank/kuntai/code/video-compression/videos/driving_0', 0),
    # ('/tank/kuntai/code/video-compression/videos/driving_1', 1),
    # ('/tank/kuntai/code/video-compression/videos/driving_2', 2),
    # ('/tank/kuntai/code/video-compression/videos/driving_3', 3),
    # ('/tank/kuntai/code/video-compression/videos/driving_4', 4),
    ('/datamirror/kuntai/accmpeg/videos/dashcamcropped_1/', 1)
]

segment_length = 30

for v, idx in video_list:
    
    length = len(glob.glob(v + '/*.png'))
    path = Path('videos/dashcamcropped/dashcamcropped_%d_3xdownsample' % idx)
    os.system(f'rm -r {path}')
    path.mkdir(parents=True)
    
    for time in range((length//segment_length) - 1):
        
        start = time * segment_length

        # perform 3x downsampling        
        run([
            'ffmpeg', '-r', '30', '-start_number', f'{start}', '-i', v + '/%010d.png', '-c:v', 'libx264', '-qp', '0', '-r', '10', '-t', '1', str(path / ('part%d.mp4' % time))
        ])

    