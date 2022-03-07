from collections import OrderedDict

import torch
import torchvision.transforms.functional as F
import yaml
from PIL import Image

from . import carn_m


class CARN:
    def __init__(self, upscale=2):
        self.net = carn_m.Net(multi_scale=True, group=4)
        self.net.eval()
        self.upscale = upscale
        print("carm m!!!!")
        state_dict = torch.load("/data/yuhanl/diff4/CARN-pytorch/checkpoint/carn_m.pth")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            # name = k[7:] # remove "module."
            new_state_dict[name] = v

        self.net.load_state_dict(new_state_dict)
        self.net.to('cuda:1')

    def __call__(self, image):
        # import pdb; pdb.set_trace()
        image = self.net(image, self.upscale)
        # print(image)
        # input()
        return image
