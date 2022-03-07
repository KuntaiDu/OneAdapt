import torch
from PIL import Image
import numpy as np
from skimage import color
from skimage import io
import subprocess
import os

import cv2 as cv
#
# a = torch.ones((6, 7))
# res = a.sum(dim=1)
# array = np.arange(0, 460800, 1, np.uint8)
# array = np.reshape(array, (720, 640))
# data = Image.fromarray(array)
# for file in os.listdir("frames"):
#     file_name = file.split(".png")[0]
#     subprocess.run(["tar", "cfz", "compressed_frames/{}.zip".format(file_name), "frames/{}".format(file)]  )
#
# avg_size = 0
# for file in os.listdir("frames"):
#     avg_size += os.path.getsize(os.path.join("compressed_frames", file.split(".png")[0] + '.zip'))
# print(avg_size/len(os.listdir('frames')))
for index in range(0, 31, 3):
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-i", f"images/compress{index}_*", "-s", "1080:720", "-c:v", "libx264", "compressed_videos/out{}.mp4".format(index)]  )
