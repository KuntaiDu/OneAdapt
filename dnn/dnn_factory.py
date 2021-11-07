"""
    The factory to build DNN according to 1). the name or 2). the yml file
"""

from dnn.efficient_det.interface import EfficientDet

from .coco_model import COCO_Model
from .dnn import DNN
from .efficient_det.interface import EfficientDet
# from .fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
# from .fcn_resnet50 import FCN_ResNet50
# from .mobilenet import SSD

# from .detr_resnet101 import Detr_ResNet101
# from .segmentation import Segmentation
# from .yolo5 import Yolo5s


class DNN_Factory:
    def __init__(self):
        self.name2model = {
            "EfficientDet": EfficientDet,
            # "MobileNet-SSD": SSD,
            # "Detr_ResNet101": Detr_ResNet101,
            # "Yolo5s": Yolo5s,
        }

    def get_model(self, name, load_model=True):


        for key in self.name2model:
            if key in name:
                return self.name2model[key](name, load_model)

        # elif "Segmentation" in name:
        #     return Segmentation(name)
        # elif name == "fcn_resnet50":
        #     return FCN_ResNet50()
        assert "yaml" in name
        return COCO_Model(name)
