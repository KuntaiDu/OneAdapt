
import sys
import os
sys.path.append(os.getcwd())
os.environ['SETTINGS_FILE'] = 'settings_encoding.toml'

from munch import *

from itertools import product
import coloredlogs

from utils.inference import inference, examine

from dnn.dnn_factory import DNN_Factory
import pymongo
from tqdm import tqdm
import yaml
from config import settings
import logging


gt_config = munchify(settings.ground_truths_config.to_dict())
fmts = [
    f'/dataheart/dataset/downtown/downtown_{i}/part%d.mp4' for i in range(1)
]
db = pymongo.MongoClient("mongodb://localhost:27017/")[settings.collection_name]
app = DNN_Factory().get_model(gt_config.app)



if __name__ == "__main__":
    
    coloredlogs.install(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s",
        level="INFO",
    )
    
    logger = logging.getLogger("convexity")
    
    for idx, fmt in enumerate(fmts):
        
        for qpidx, qp in enumerate([22, 24, 28, 32, 36]):
            for res in range(4,13,2):
                
                for sec in range(10):
                
                    gt_args = gt_config.copy()
                    gt_args.update({
                        "input": fmt, 
                        "second": sec,
                        "approach": "mpeg"
                    })
                    
                    x_args = gt_args.copy()
                    x_args.qp = qp
                    x_args.res = f'{int(res*100*1.6)}x{res*100}'
                    
                    x = examine(x_args, gt_args, app, db)
                    
                    with open('stats/measurement_convexity_new2.yaml', 'a') as f:
                        f.write(yaml.dump([{
                            'f1': x['f1'],
                            'bw': x['my_bw'],
                            'qp': qp,
                            'qpidx': qpidx,
                            'res': res,
                            'videoid': idx,
                        }]))