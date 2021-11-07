
import av
import torch
import torchvision.transforms as T
from pathlib import Path
import yaml


def read_video(video_name):

    container = av.open(video_name)
    frames = []
    fid = 0

    for fid, frame in enumerate(container.decode(video=0)):
        yield (fid, T.ToTensor()(frame.to_image()).unsqueeze(0))
        

# def read_video_frame_numbers(video_name, args):

#     container = av.open(video_name)
#     return container.streams.video[0].frames

def read_video_config(video_name):

    if Path(video_name).exists():
        container = av.open(video_name)

        return {
            '#frames': container.streams.video[0].frames,
            'bw': Path(video_name).stat().st_size,
        }
    else:
        raise FileNotFoundError("%s does not exist, can't parse the config of it." % video_name)
        