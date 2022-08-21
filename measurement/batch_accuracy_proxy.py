import sys
import os
sys.path.append('/datamirror/kuntai/code/diff')
os.environ['SETTINGS_FILE'] = 'measurement/settings_macroblock.toml'
from config import settings
from utils.inference import examine, inference
from dnn.dnn_factory import DNN_Factory
from utils.encode import encode
from utils.video_reader import read_video_to_tensor
from tqdm import tqdm
import torch.nn.functional as F
from munch import munchify
import torch
import pymongo
import yaml


db = pymongo.MongoClient("mongodb://localhost:27017/")[settings.collection_name]



app = DNN_Factory().get_model('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')

# pick a random point and calculate saliency
def get_saliency(args):

    video_name, video_config = encode(args)
    video = read_video_to_tensor(video_name)
    saliencies = []

    for idx, frame in enumerate(tqdm(video)):

        frame = frame.unsqueeze(0)
        result = app.inference(frame, detach=False, grad=False)
        result = app.filter_result(result)
        frame = frame.clone()
        frame.requires_grad = True
        loss = app.calc_loss(frame, result)
        loss.backward()
        saliencies.append(frame.grad)

    return video, torch.cat(saliencies)


macroblocks = [F.interpolate(torch.randint(24, 37, [5, 8]).float()[None,None,:,:], [45, 80])[0, 0].int() for i in range(10)]


if __name__ == '__main__':

    # my_video, my_saliency = get_saliency(my_config)
    os.system("rm temp.yaml")
    dproxies = []
    daccs= []
    dabss = []
    dmses = []
    for second in tqdm(range(80, 86)):

        
        my_config = munchify({
            'input': 'videos/driving/driving_0/part%d.mp4',
            'second': second,
            'approach': 'mpeg',
            'macroblocks': 30 * torch.ones([45, 80]),
            'fr': 10,
            'res': '1280x720',
            'app': 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml',
        })
        gt_config = munchify({
            'input': 'videos/driving/driving_0/part%d.mp4',
            'second': second,
            'approach': 'mpeg',
            'macroblocks': 24 * torch.ones([45, 80]),
            'fr': 10,
            'res': '1280x720',
            'app': 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml',
        })
        my_video, my_saliency = get_saliency(my_config)
        gt_video = read_video_to_tensor(encode(gt_config)[0])
        my_accuracy = examine(my_config, gt_config, app, db)['f1']

        x = []
        x1 = []
        x2 = []
        y = []
        for mb in tqdm(macroblocks):
            cur_config = my_config.copy()
            cur_config['macroblocks'] = mb
            cur_video = read_video_to_tensor(encode(cur_config)[0])
            cur_accuracy = examine(cur_config, gt_config, app, db)['f1']
            delta_acc = cur_accuracy - my_accuracy
            delta_proxy = ((my_saliency.abs() * (my_video - gt_video).abs()).sum() -\
                (my_saliency.abs() * (cur_video - gt_video).abs()).sum()).item()
            delta_abs = ((my_video - gt_video).abs().sum() - (cur_video - gt_video).abs().sum()).item()
            delta_mse = ((my_video - gt_video) ** 2).sum() - ((cur_video - gt_video)**2).sum()
            x.append(delta_proxy)
            x1.append(delta_abs)
            x2.append(delta_mse)
            y.append(delta_acc)
        dproxies.append(x)
        daccs.append(y)
        dabss.append(x1)
        dmses.append(x2)

    with open('temp_metrics.yaml', 'w') as f:
        dproxies = torch.tensor(dproxies)
        daccs = torch.tensor(daccs)
        dmses = torch.tensor(dmses)
        dabss = torch.tensor(dabss)
        f.write(yaml.dump((dproxies, daccs, dmses, dabss)))
            
    