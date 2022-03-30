import logging

import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from utils.bbox_utils import *
from utils.visualize_mask import InstSegVisualization
from .dnn import DNN
import numpy as np
from utils.configure import Config


class MaskRCNN_ResNet50_FPN(DNN):
    def __init__(self):

        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.name = 'MaskRCNN_ResNet50_FPN'
        self.cfg = Config()
        self.cfg.update(['maskrcnn.yaml'])
        self.logger = logging.getLogger(self.name)
        handler = logging.NullHandler()
        self.logger.addHandler(handler)
    def name(self):
        return self.name
    def cpu(self):

        self.model.cpu()
        self.logger.info(f"Place {self.name()} on CPU.")

    def cuda(self):

        self.model.cuda()
        self.logger.info(f"Place {self.name()} on GPU.")

    def inference(self, video, detach=False, grad=False, dryrun = False, feature=False):

        assert len(video.shape) == 4, "The video tensor should be 4D"

    
        if grad:
            context = torch.enable_grad()
            # video.requires_grad = True
        else:
            context = torch.no_grad()

        with context:
            res = self.model(video)
            # print(res)
        return res

    def visualize(self, image, result):
        # set_trace()
        # result = self.filter_result(result, args, gt=gt)
        v = Visualizer(image, MetadataCatalog.get("coco_2017_train"), scale=1)
        out = v.draw_instance_predictions(result["instances"])
        return Image.fromarray(out.get_image(), "RGB")
    def visualize_results(self, target, image, file_name):
        # print(image.shape)
        v = InstSegVisualization(
                    self.cfg, image=image[0],
                    boxes=target['boxes'], labels=target['labels'],
                    masks=target['masks'])
        v.plot_image()
        v.add_bbox()
        # v.add_label()
        # v.add_binary_mask()
        v.save(file_name)
    def calculate_iou_mask(self, mask1, mask2):
        iou_score = torch.zeros((mask1.shape[0], mask2.shape[0]))
        for i in range(mask1.shape[0]):
            for j in range(mask2.shape[0]):
                intersection = torch.logical_and(mask1[i][0], mask2[j][0])
                union = torch.logical_or(mask1[i][0], mask2[j][0])
                iou_score[i][j] = torch.sum(intersection).item() / torch.sum(union).item()
        
        return iou_score

    def calc_accuracy(self, video, gt, args, images):
        """
            Calculate the accuracy between video and gt using thresholds from args based on inference results
        """

        assert video.keys() == gt.keys()
        args.confidence_threshold = 0.2
        args.gt_confidence_threshold = 0.2
        args.iou_threshold = 0.2
        accuracies = []
        for fid in video.keys():
            video_res = video[fid][0]
            gt_res = gt[fid][0]
            self.visualize_results(video_res, images[fid], 'temp.jpg')
            video_scores = video_res["scores"]
            video_ind = video_scores > args.confidence_threshold
            video_bboxes = video_res["boxes"][video_ind, :]
            video_masks = video_res["masks"][video_ind, :]
            video_labels = video_res["labels"][video_ind]

            gt_scores = gt_res["scores"]
            gt_ind = gt_scores > args.confidence_threshold
            gt_bboxes = gt_res["boxes"][gt_ind, :]
            gt_masks = gt_res["masks"][gt_ind, :]
            gt_labels = gt_res["labels"][gt_ind]
            # self.visualize_results(gt_res, images[fid], 'gt_temp.jpg')
            IoU = self.calculate_iou_mask(video_masks, gt_masks)

            # let IoU = 0 if the label is wrong
            fat_video_labels = video_labels[:, None].repeat(1, len(gt_labels))
            fat_gt_labels = gt_labels[None, :].repeat(len(video_labels), 1)
            IoU[fat_video_labels != fat_gt_labels] = 0

            # calculate f1
            tp, fp, fn = 0, 0, 0
            # print(IoU)
            for i in range(len(gt_labels)):
                if (IoU[:, i] > args.iou_threshold).sum() > 0:
                    tp += 1
                else:
                    fn += 1
            fp = len(video_labels) - tp

            f1 = None
            if fp + fn == 0:
                f1 = 1
            else:
                f1 = 2 * tp / (2 * tp + fp + fn)
            self.logger.info(f"Get an recall score {tp/(tp+fn)} at frame {fid}")
            self.logger.info(f"Get an precision score {tp/(tp+fp)} at frame {fid}")

            self.logger.info(f"Get an f1 score {f1} at frame {fid}")

            accuracies.append(f1)

        return torch.tensor(accuracies).mean()

    def calc_loss(self, video, gt, args):
        """
            Inference and calculate the loss between video and gt using thresholds from args
        """

        assert (
            video.shape == gt.shape
        ), f"The shape of video({video.shape}) and gt({gt.shape}) must be the same in order to calculate the loss"
        assert len(video.shape) == 4, f"The shape of video({video.shape}) must be 4D."

        # inference, and obtain the inference results
        self.model.eval()
        gt_results = self.inference(gt)[0]
        gt_scores = gt_results["scores"]
        gt_ind = gt_scores > args.confidence_threshold
        gt_bboxes = gt_results["boxes"][gt_ind, :]
        gt_masks = gt_results["masks"][gt_ind, :]
        gt_labels = gt_results["labels"][gt_ind]

        # construct targets
        targets = [{"boxes": gt_bboxes, "labels": gt_labels, "masks": gt_masks}]

        # switch the model to training mode to obtain loss
        self.model.train()
        self.model.zero_grad()
        assert (
            self.is_cuda and video.is_cuda
        ), "The video tensor and the model must be placed on GPU to perform inference"
        with torch.enable_grad():
            losses = self.model(video, targets)

        return sum(loss for loss in losses.values())
