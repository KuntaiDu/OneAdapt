
import torch
import torch.nn.functional as F
from pdb import set_trace
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as T
import io
import PIL

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

def dilate_binarize(mask, lower_bound, kernel_size, cuda=True):
    kernel = torch.ones([1, 1, kernel_size, kernel_size])
    if cuda:
        kernel = kernel.cuda(non_blocking=True)
    mask = torch.where(
        (mask > lower_bound), torch.ones_like(mask), torch.zeros_like(mask),
    )
    mask = F.conv2d(mask, kernel, stride=1, padding=(kernel_size - 1) // 2,)
    mask = torch.where(
        mask > 0.5, torch.ones_like(mask), torch.zeros_like(mask),
    )
    return mask




class framerate_control:

    def __init__(self, framerates):

        assert framerates == sorted(framerates)

        self.framerates = framerates

        self.compute = torch.tensor([1.])
        self.compute.requires_grad = True

        # self.fr_id = 0
        # self.weights = [torch.tensor([0.]) for i in self.framerates]
        # for i in range(len(self.framerates)):
        #     if self.fr_id != i:
        #         self.weights[i].requires_grad = True
        # self.weights[self.fr_id] = 1 - sum(self.weights)
        
        self.logger = logging.getLogger("framerate_control")


    def step(self):
        self.compute.requires_grad = False
        print(self.compute.grad)
        self.compute = self.compute - 0.01 * self.compute.grad
        if self.compute > 1:
            self.compute[:] = 1
        if self.compute < 1/self.framerates[-1]:
            self.compute[:] = 1/self.framerates[-1] + 0.0001
        
        self.compute.requires_grad = True

        self.logger.info('New compute: %f', self.compute)

    def get_compute(self):

        return self.compute

    def __call__(self, images):

        assert isinstance(images, torch.Tensor)


        def transform(fr):

            split_images = images.split(fr)

            def tile_images(x):
                return torch.cat([x[0].unsqueeze(0) for i in range(len(x))])

            split_images = torch.cat([tile_images(i) for i in split_images])

            return split_images


        images = [transform(fr) for fr in self.framerates]

        lfr,rfr = None, None
        l, r = None, None

        for i in range(len(self.framerates)-1):
            if 1/self.framerates[i] >= self.compute >= 1/self.framerates[i+1]:
                lfr, rfr = 1/self.framerates[i], 1/self.framerates[i+1]
                l, r = i, i+1
                break

        weight = (self.compute - rfr) / (lfr-rfr)
        return weight * images[l] + (1-weight) * images[r]



# class framerate_control:

#     def __init__(self, framerates):

#         self.framerates = framerates

#         self.fr_id = 0
#         self.weights = [torch.tensor([0.]) for i in self.framerates]
#         for i in range(len(self.framerates)):
#             if self.fr_id != i:
#                 self.weights[i].requires_grad = True
#         self.weights[self.fr_id] = 1 - sum(self.weights)
        
#         # self.weights = torch.zeros([len(self.framerates)])
#         # self.weights[0] = 1
#         # self.weights.requires_grad = True
#         # self.cum_grad = torch.zeros_like(self.weights)
        
#         self.logger = logging.getLogger("framerate_control")


#     def step(self):
#         self.weights[self.fr_id].grad = torch.tensor([0.])
#         for i in range(len(self.framerates)):
#             if self.weights[i].grad < self.weights[self.fr_id].grad:
#                 self.fr_id = i
#         self.weights = [torch.tensor([0.]) for i in self.framerates]
#         for i in range(len(self.framerates)):
#             if self.fr_id != i:
#                 self.weights[i].requires_grad = True
#         self.weights[self.fr_id] = 1 - sum(self.weights)
#         # self.cum_grad = self.cum_grad * 0.2 + self.weights.grad * 0.8
#         # self.weights.requires_grad = False
#         # self.weights[:] = 0
#         # self.weights[self.cum_grad.argmin()] = 1
#         # self.weights.requires_grad = True

#         self.logger.info('New frame rate: %d', self.framerates[self.fr_id])
        

#     def get_compute(self):

#         return sum(weight * (1/ fr) for weight, fr in zip(self.weights, self.framerates))

#     def __call__(self, images):

#         assert isinstance(images, torch.Tensor)


#         def transform(fr):

#             split_images = images.split(fr)

#             def tile_images(x):
#                 return torch.cat([x[0].unsqueeze(0) for i in range(len(x))])

#             split_images = torch.cat([tile_images(i) for i in split_images])

#             return split_images


#         images = [transform(fr) for fr in self.framerates]

#         return sum(weight * image for (weight, image) in zip(self.weights, images))


class quality_control:

    def __init__(self, writer, thresh = 5e8):

        self.q = torch.ones([1,1,9,16])
        self.q.requires_grad = True
        # self.cum_grad = torch.zeros_like(self.q)
        self.logger = logging.getLogger("quality_control")
        self.writer = writer
        self.thresh = thresh

    def step(self):
        # self.cum_grad = self.cum_grad * 0.2 + self.q.grad * 0.8
        self.grad = self.q.grad.abs()
        self.q.requires_grad = False
        self.q[:, :, :, :] = dilate_binarize((self.grad > self.thresh).float(), 0.5, 3, False)
        self.q.requires_grad = True
        self.logger.info('New mean qulaity: %.3f', self.q.mean())

        self.q.grad.zero_()


    def __call__(self, hq, lq):

        q = tile_mask(self.q, 80)
        return hq * q + lq * (1-q)

    def get_size(self):

        return 0.35 + 0.65 * self.q.mean()

    def visualize(self, image, fid):

        self.writer.add_image('image', image, fid)

        fig, ax = plt.subplots(1, 1, figsize=(11,5), dpi=200)
        heat = self.grad.detach()
        heat = tile_mask(heat, 80)[0, 0, :, :]
        ax = sns.heatmap(
            heat.numpy(),
            zorder=3,
            alpha=0.5,
            ax=ax,
            xticklabels=False,
            yticklabels=False
        )
        ax.imshow(T.ToPILImage()(image.detach()), zorder=3, alpha = 0.5)
        ax.tick_params(left=False, bottom=False)
        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches="tight")
        buf.seek(0)
        result = PIL.Image.open(buf)
        self.writer.add_image('heat', T.ToTensor()(result), fid)
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(11,5), dpi=200)
        heat = self.q.detach()
        heat = tile_mask(heat, 80)[0, 0, :, :]
        ax = sns.heatmap(
            heat.numpy(),
            zorder=3,
            alpha=0.5,
            ax=ax,
            xticklabels=False,
            yticklabels=False
        )
        ax.imshow(T.ToPILImage()(image.detach()), zorder=3, alpha = 0.5)
        ax.tick_params(left=False, bottom=False)
        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches="tight")
        buf.seek(0)
        result = PIL.Image.open(buf)
        self.writer.add_image('quality', T.ToTensor()(result), fid)
        plt.close()
        



# class resolution_control:

#     def __init__(self, resolutions, cur_resolution):

#         self.resolutions = resolutions
#         self.cur_resolution = cur_resolution
#         self.weights = [torch.tensor(0.) if res != self.cur_resolution else torch.tensor(1.) for res in self.resolutions]

#         for i in self.weights:
#             i.requires_grad = True

#         assert cur_resolution in resolutions

#     def __call__(self, images):

#         assert isinstance(images, torch.Tensor)

#         images = [F.interpolate(F.interpolate(images, size = res), size=(720, 1280)) for res in self.resolutions]

#         return sum(weight * image for (weight, image) in zip(self.weights, images))

    

        

        

        

        