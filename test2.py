import torch
from PIL import Image
import numpy as np
from skimage import color
from skimage import io
import subprocess
import os
import torch.nn.functional as F


import cv2 as cv
#


def tile_mask(mask, tile_size):
    """
        Here the mask is of shape [1, 1, H, W]
        Eg:
        mask = [    1   2
                    3   4]
        tile_mask(mask, 2) will return
        ret =  [    1   1   2   2
                    1   1   2   2
                    3   3   4   4
                    3   3   4   4]
        This function controlls the granularity of the mask.
    """
    mask = mask[0, 0, :, :]
    t = tile_size
    mask = mask.unsqueeze(1).repeat(1, t, 1).view(-1, mask.shape[1])
    mask = mask.transpose(0, 1)
    mask = mask.unsqueeze(1).repeat(1, t, 1).view(-1, mask.shape[1])
    mask = mask.transpose(0, 1)
    return torch.cat(3 * [mask[None, None, :, :]], 1)


x = torch.ones((1, 1, 720, 1280)) * 10
kernel = torch.ones([1, 1, 16, 16])
saliency = F.conv2d(x, kernel, stride=16, padding=0)
saliency = torch.where(saliency > 2000, torch.ones(saliency.shape), torch.zeros(saliency.shape))

y  = tile_mask(saliency, 16)
print(y.shape)
print(y)
