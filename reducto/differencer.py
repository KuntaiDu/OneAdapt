
import configparser

import cv2
import imutils
import numpy as np



class PixelDiff:

    def __init__(self, thresh=.0, fraction=.0, dataset=None):

        self.feature = 'pixel'
        self.pixel_thresh_low_bound = 21
        
    def get_frame_feature(self, frame):
        return frame

    def cal_frame_diff(self, frame, prev_frame):
        total_pixels = frame.shape[0] * frame.shape[1]
        frame_diff = cv2.absdiff(frame, prev_frame)
        frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.threshold(frame_diff, self.pixel_thresh_low_bound,
                                   255, cv2.THRESH_BINARY)[1]
        changed_pixels = cv2.countNonZero(frame_diff)
        fraction_changed = changed_pixels / total_pixels
        return fraction_changed


class AreaDiff:

    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        self.area_blur_rad = 11
        self.area_blur_var = 0
        self.area_thresh_low_bound = 21
        self.feature = 'area'

    def get_frame_feature(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (self.area_blur_rad, self.area_blur_rad),
                                self.area_blur_var)
        return blur

    def cal_frame_diff(self, frame, prev_frame):
        total_pixels = frame.shape[0] * frame.shape[1]
        frame_delta = cv2.absdiff(frame, prev_frame)
        thresh = cv2.threshold(frame_delta, self.area_thresh_low_bound, 255,
                               cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None)
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        if not contours:
            return 0.0
        return max([cv2.contourArea(c) / total_pixels for c in contours])
       


class EdgeDiff:

    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        self.edge_blur_rad = 11
        self.edge_blur_var = 0
        self.edge_canny_low = 101
        self.edge_canny_high = 255
        self.edge_thresh_low_bound = 21
        self.feature = 'edge'

    def get_frame_feature(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (self.edge_blur_rad, self.edge_blur_rad),
                                self.edge_blur_var)
        edge = cv2.Canny(blur, self.edge_canny_low, self.edge_canny_high)
        return edge

    def cal_frame_diff(self, edge, prev_edge):
        total_pixels = edge.shape[0] * edge.shape[1]
        frame_diff = cv2.absdiff(edge, prev_edge)
        frame_diff = cv2.threshold(frame_diff, self.edge_thresh_low_bound, 255,
                                   cv2.THRESH_BINARY)[1]
        changed_pixels = cv2.countNonZero(frame_diff)
        fraction_changed = changed_pixels / total_pixels
        return fraction_changed
        


class CornerDiff:


    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        self.feature = 'corner'
        self.corner_block_size = 5
        self.corner_ksize = 3
        self.corner_k = 0.05

    def get_frame_feature(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corner = cv2.cornerHarris(gray, self.corner_block_size,
                                  self.corner_ksize, self.corner_k)
        corner = cv2.dilate(corner, None)
        return corner

    def cal_frame_diff(self, corner, prev_corner):
        total_pixels = corner.shape[0] * corner.shape[1]
        frame_diff = cv2.absdiff(corner, prev_corner)
        changed_pixels = cv2.countNonZero(frame_diff)
        fraction_changed = changed_pixels / total_pixels
        return fraction_changed

        


class HistDiff:

    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        self.feature = 'histogram'
        self.hist_nb_bins = 32

    def get_frame_feature(self, frame):
        nb_channels = frame.shape[-1]
        hist = np.zeros((self.hist_nb_bins * nb_channels, 1), dtype='float32')
        for i in range(nb_channels):
            hist[i * self.hist_nb_bins: (i + 1) * self.hist_nb_bins] = \
                cv2.calcHist(frame, [i], None, [self.hist_nb_bins], [0, 256])
        hist = cv2.normalize(hist, hist)
        return hist

    def cal_frame_diff(self, frame, prev_frame):
        return cv2.compareHist(frame, prev_frame, cv2.HISTCMP_CHISQR)

        


class HOGDiff:


    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        self.feature = 'HOG'
        self.hog_resize = 512
        self.hog_orientations = 10
        self.hog_pixel_cell = 5
        self.hog_cell_block = 2

    def get_frame_feature(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize to speed up
        gray = cv2.resize(gray, (self.hog_resize, self.hog_resize))
        hog = feature.hog(gray, orientations=self.hog_orientations,
                          pixels_per_cell=(self.hog_pixel_cell, self.hog_pixel_cell),
                          cells_per_block=(self.hog_cell_block, self.hog_cell_block)
                          ).astype('float32')
        return hog

    def cal_frame_diff(self, frame, prev_frame):
        dis = np.linalg.norm(frame - prev_frame)
        dis /= frame.shape[0]
        return dis

        


class SIFTDiff:


    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        self.feature = 'SIFT'

    def get_frame_feature(self, frame):
        sift = cv2.xfeatures2d.SIFT_create()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, des = sift.detectAndCompute(gray, None)
        des = (np.mean(des, axis=0).astype('float32')
               if des is not None else np.zeros(128))
        return des

    def cal_frame_diff(self, frame, prev_frame):
        dis = np.linalg.norm(frame - prev_frame)
        dis /= frame.shape[0]
        return dis


class SURFDiff:

    feature = 'SURF'

    def __init__(self, thresh=.0, fraction=.0, dataset=None):
        self.feature = 'SURF'
        self.surf_hessian_thresh = 400

    def get_frame_feature(self, frame):
        surf = cv2.xfeatures2d.SURF_create()
        surf.setUpright(True)
        surf.setHessianThreshold(self.surf_hessian_thresh)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, des = surf.detectAndCompute(gray, None)
        des = np.zeros(128) if des is None else np.mean(des, axis=0).astype('float32')
        return des

    def cal_frame_diff(self, frame, prev_frame):
        dis = np.linalg.norm(frame - prev_frame)
        dis /= frame.shape[0]
        return dis
        



reducto_differencers = [
    PixelDiff(), 
    AreaDiff(), 
    EdgeDiff(), 
    CornerDiff(),
    HistDiff(), 
    HOGDiff(),
    SIFTDiff(),
    SURFDiff()]