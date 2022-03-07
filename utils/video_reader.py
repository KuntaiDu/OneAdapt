import av
import torch
import torchvision.transforms as T
from pathlib import Path
import yaml
from config import settings
import torch.nn.functional as F


def read_video(video_name):

    container = av.open(video_name)
    container.streams.video[0].thread_type = 'AUTO'
    frames = []
    fid = 0

    for fid, frame in enumerate(container.decode(video=0)):
        yield (fid, T.ToTensor()(frame.to_image()).unsqueeze(0))

def augment(result, lengt):
    assert len(result) <= lengt

    factor = (lengt + (len(result) - 1)) // len(result)

    return torch.cat([result[i // factor][None, :, :, :] for i in range(lengt)])



def read_video_to_tensor(video_name):

    video = list(read_video(video_name))
    video = torch.cat([i[1] for i in video])
    video = augment(video, settings.segment_length)
    video = F.interpolate(video, settings.input_shape)
    return video

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
        
