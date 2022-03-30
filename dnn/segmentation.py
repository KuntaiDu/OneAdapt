import logging
from pdb import set_trace

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from PIL import Image
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    fcn_resnet50,
    fcn_resnet101,
)
from utils.bbox_utils import *

from .dnn import DNN

COCO_NAMES = [
    "__background__",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [1,2,3]

    # Two dimensional
    elif len(shape) == 4 : return [1,2]

    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')

label_colors = [
    (0, 0, 0),  # 0=background
    # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
    (0, 128, 128),
    (128, 128, 128),
    (64, 0, 0),
    (192, 0, 0),
    (64, 128, 0),
    # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
    (192, 128, 0),
    (64, 0, 128),
    (192, 0, 128),
    (64, 128, 128),
    (192, 128, 128),
    # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
    (0, 64, 0),
    (128, 64, 0),
    (0, 192, 0),
    (128, 192, 0),
    (0, 64, 128),
]

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricFocalLoss(nn.Module):
    """For Imbalanced datasets
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.25
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, delta=0.25, gamma=2., epsilon=1e-07):
        super(AsymmetricFocalLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):

        axis = identify_axis(y_true.size())
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        cross_entropy = -y_true * torch.log(y_pred)
        new_ce = torch.Tensor(cross_entropy.shape)
        # print(cross_entropy.shape)
	# Calculate losses separately for each class, only suppressing background class
        back_ce = torch.pow(1 - y_pred[:,0, :,:], self.gamma) * cross_entropy[:,0,:,:]

        back_ce =  (1 - self.delta) * back_ce
        new_ce[:, 0, :,:] = back_ce
        fore_ce = cross_entropy[:, 1:, :, :]
        fore_ce = self.delta * fore_ce
        new_ce[:,1:,:,:] =  fore_ce
        loss = torch.sum(torch.sum(new_ce, axis=-1))
        return loss
class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(
            input, target, reduction=self.reduction, weight=self.weight
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class Segmentation(DNN):
    def __init__(self, name):

        model_name = name.split("/")[-1]
        exec(f"self.model = {model_name}(pretrained=True)")
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.name = name

        self.logger = logging.getLogger(self.name)
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        # self.class_ids = [2, 6, 7, 14, 15]
        self.class_ids = [0, 7]

        self.transform = T.Compose(
            [T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        self.metadata = MetadataCatalog.get("my_fcn_resnet50")
        self.metadata.stuff_classes = COCO_NAMES
        self.metadata.stuff_colors = label_colors
        self.focal_loss = AsymmetricFocalLoss()
        self.is_cuda = False

    def cpu(self):

        self.model.cpu()
        self.is_cuda = False
        self.logger.info(f"Place {self.name} on CPU.")

    def cuda(self):

        self.model.cuda()
        self.is_cuda = True
        self.logger.info(f"Place {self.name} on GPU.")

    # def parallel(self, local_rank):
    #     self.model = torch.nn.parallel.DistributedDataParallel(
    #         self.model, device_ids=[local_rank], find_unused_parameters=True
    #     )

    def inference(self, video, grad = False, detach=False, raw=False):
        """
            Generate inference results. Will put results on cpu if detach=True.
        """

        self.model.eval()
        if not self.is_cuda:
            self.cuda()
        if not video.is_cuda:
            video = video.cuda()
        if grad:
            context = torch.enable_grad()
            # video.requires_grad = True
        else:
            context = torch.no_grad()
        video = F.interpolate(video, size=(720, 1280))
        video = torch.cat([self.transform(v) for v in video.split(1)])

        with context:
            results = self.model(video)
        # print(results['aux'].shape)

        results = results["out"]
        """ newly added here"""

        results = results[:, self.class_ids, :, :]
        results_raw = results
        results = results.argmax(1).byte()
        #
        # for i in range(len(results[0])):
        #     for j in range(len(results[0][0])):
        #         if results_prob[0][i][j] > 0:
        #             print(results_prob[0][i][j])

        if detach:
            results = results.detach().cpu()
        if raw:
            return results_raw

        return results

    def transform_output(self, gt_result):
        output = torch.zeros((1, len(self.class_ids), gt_result[0].shape[0], gt_result[0].shape[1])).cuda()
        x = torch.ones((gt_result[0].shape[0], gt_result[0].shape[1])).cuda()
        y = torch.zeros((gt_result[0].shape[0], gt_result[0].shape[1])).cuda()
        # for i in range(len(gt_result[0])):
        #     for j in range(len(gt_result[0][0])):
        #         if gt_result[0][i][j] > 0:
        #             print(gt_result[0][i][j])
        for i in range(len(self.class_ids)):
            output[0][i] = torch.where(gt_result[0] == i, x ,y)
            # print(torch.sum(output[0][i] == 1))
        return output

    # def dice_loss(self, input, target):
    #     smooth = 1.
    #     input = input[0, 1:, :,:]
    #     target = target[0, 1:, :, :]
    #     print(input.shape)
    #     iflat = input.view(-1)
    #     tflat = target.view(-1)
    #     intersection = (iflat * tflat).sum()
    #
    #     return - ((2. * intersection + smooth) /
    #               (iflat.sum() + tflat.sum() + smooth))

    def jaccard_loss(self, logits, true, eps=1e-7):
        """Computes the Jaccard loss, a.k.a the IoU loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the jaccard loss so we
        return the negated jaccard loss.
        Args:
            true: a tensor of shape [B, H, W] or [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            jacc_loss: the Jaccard loss.
        """
        # logits = logits[:, 1:, :, :]
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1).to(torch.int64)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        # print(torch.sum(true_1_hot[:, 0, :,:]))
        probas = probas[:, 1:, :, :]
        true_1_hot = true_1_hot[:, 1:, :, :]
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        union = cardinality - intersection
        jacc_loss = (intersection / (union + eps)).mean()
        return (1 - jacc_loss)

    def dice_loss(self, input, target):
        """
        input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
        target is a 1-hot representation of the groundtruth, shoud have same size as the input
        """
        assert input.size() == target.size(), "Input sizes must be equal."
        assert input.dim() == 4, "Input must be a 4D Tensor."
        uniques=np.unique(target.cpu().numpy())
        assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"

        probs=F.softmax(input)
        num=probs*target#b,c,h,w--p*g
        num=torch.sum(num,dim=3)#b,c,h
        num=torch.sum(num,dim=2)


        den1=probs*probs#--p^2
        den1=torch.sum(den1,dim=3)#b,c,h
        den1=torch.sum(den1,dim=2)


        den2=target*target#--g^2
        den2=torch.sum(den2,dim=3)#b,c,h
        den2=torch.sum(den2,dim=2)#b,c


        dice=2*(num/(den1+den2))
        dice_eso=dice[:,1:]#we ignore bg dice val, and take the fg

        dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz

        return dice_total

    def step(self, tensor):
        return (10 * tensor).sigmoid()

    def filter_result(self, video_results, args):

        """
        BYPASS THIS FUNCTION
        """

        return video_results

        from skimage import measure

        bin_video = torch.where(
            video_results > 0,
            torch.ones_like(video_results),
            torch.zeros_like(video_results),
        )

        # set_trace()

        bin_video = torch.tensor(
            measure.label(bin_video.numpy().astype(int)), dtype=torch.int32
        )
        nclass = torch.max(bin_video).item()
        mask = torch.zeros_like(video_results)

        for i in range(1, nclass + 1):
            size = torch.sum(bin_video == i) * 1.0 / bin_video.numel()
            if size < args.size_bound:
                mask[bin_video == i] = 1

        return video_results * mask

    def calc_accuracy(self, video, gt, args):
        """
            Calculate the accuracy between video and gt using thresholds from args based on inference results
        """

        assert video.keys() == gt.keys()

        accs = []

        for fid in video.keys():

            if fid % 10 == 0:
                print(fid)

            video_result = self.filter_result(video[fid], args)
            gt_result = self.filter_result(gt[fid], args)

            mask = ~((video_result == 0) & (gt_result == 0))
            correct = (video_result == gt_result) & mask

            ncorrect = len(correct.nonzero(as_tuple=False))
            nall = len(mask.nonzero(as_tuple=False))
            if nall != 0:
                accs.append(ncorrect / nall)
            else:
                accs.append(1.0)

            # if fid % 10 == 9:
            #     #pass
            #     print('f1:', torch.tensor(f1s[-9:]).mean().item())
            #     print('pr:', torch.tensor(prs[-9:]).mean().item())
            #     print('re:', torch.tensor(res[-9:]).mean().item())

        return {"acc": torch.Tensor(accs).mean().item()}

    def calc_loss(self, videos, gt_results, args, train=False):
        """
            Inference and calculate the loss between video and gt using thresholds from args
        """

        if not self.is_cuda:
            self.cuda()
        if not videos.is_cuda:
            videos = videos.cuda()

        videos = F.interpolate(videos, size=(720, 1280))
        videos = torch.cat([self.transform(v) for v in videos.split(1)])

        targets = gt_results.cuda()

        # switch the model to training mode to obtain loss
        # set_trace()
        return FocalLoss(weight=torch.ones(len(COCO_NAMES)).cuda())(
            self.model(videos)["out"], targets[:, :, :].long()
        )

    def visualize(self, image, result, args):
        # set_trace()
        result = self.filter_result(result, args)
        result['instances'] = result['instances'][0]
        v = Visualizer(image, self.metadata, scale=1)
        out = v.draw_sem_seg(result['instances'])
        return Image.fromarray(out.get_image(), "RGB")

    def get_undetected_ground_truth_index(self, gt, video, args):

        (
            video_ind,
            video_scores,
            video_bboxes,
            video_labels,
        ) = self.filter_results(video, args.confidence_threshold)
        gt_ind, gt_scores, gt_bboxes, gt_labels = self.filter_results(
            gt, args.confidence_threshold
        )

        # get IoU and clear the IoU of mislabeled objects
        IoU = jaccard(video_bboxes, gt_bboxes)
        fat_video_labels = video_labels[:, None].repeat(1, len(gt_labels))
        fat_gt_labels = gt_labels[None, :].repeat(len(video_labels), 1)
        IoU[fat_video_labels != fat_gt_labels] = 0

        return (IoU > args.iou_threshold).sum(dim=0) == 0
