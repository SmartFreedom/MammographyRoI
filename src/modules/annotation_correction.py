import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import cv2
import os
import scipy

from skimage import filters
from skimage import exposure

from ..configs import config
from ..modules import enhanced as en


kalcinaty_map = {
    1: 1,
    2: 2,
    3: 3,
}


def dice(a, b):
    a = a.astype(np.bool)
    b = b.astype(np.bool)
    return ((a & b).sum() ) / (min(a.sum(), b.sum()) + 1e-5)


def merge(masks_new, masks_old, data_new, data_old, productivity):
    for cn in np.unique(masks_new['idx'][masks_new['idx'] != 0]):
        for co in np.unique(masks_old['idx'][masks_old['idx'] != 0]):
            roin = masks_new['idx'] == cn
            roio = masks_old['idx'] == co
            if dice(roin, roio) > .5:
                row_old = data_old.query('id_segment==@co')
                idx_old = row_old.index[0]
                row_old = row_old.reset_index().loc[0]

                row_new = data_new.query('id_segment==@cn')
                idx_new = row_new.index[0]
                row_new = row_new.reset_index().loc[0]

                if int(row_old.kalcinaty) > int(row_new.kalcinaty):
                    masks_new['idx'][roin] = 0
                    masks_new['class'][roin] = 0
                    data_new.drop(idx_new, inplace=True)
                elif int(row_old.kalcinaty) > int(row_new.kalcinaty):
                    masks_old['idx'][roio] = 0
                    masks_old['class'][roio] = 0
                    data_old.drop(idx_old, inplace=True)
                elif productivity[row_old['Исследователь']] > productivity[row_new['Исследователь']]:
                    masks_new['idx'][roin] = 0
                    masks_new['class'][roin] = 0
                    data_new.drop(idx_new, inplace=True)
                else:
                    masks_old['idx'][roio] = 0
                    masks_old['class'][roio] = 0
                    data_old.drop(idx_old, inplace=True)

    return (
        { 
            k: np.max([masks_new[k], masks_old[k]], axis=0) 
            for k in masks_new.keys() 
        },
        data_new.reset_index(drop=True), 
        data_old.reset_index(drop=True)
    )


def small_crop_around(image, centroid):
    centroid = np.array(centroid).astype(np.int)
    delta = (np.array(image.shape) * .005).astype(np.int)
    return image[
        max(0, centroid[0] - delta[0]): centroid[0] + delta[0],
        max(0, centroid[1] - delta[1]): centroid[1] + delta[1]
    ]


def crop_centroid(image, centroid, side=config.CROP_SIDE, centroids=[]):
    # centroids -> (y, x, id_segment, points_nb, ..)
    centroid = np.array(centroid).astype(np.int)
    centroids = np.array(centroids).astype(np.int)
    centroids[..., :2] = centroids[..., :2] - (centroid - side // 2)
    shape = np.array(image.shape)[:2]
    xy_min, xy_max = (centroid - side // 2), (centroid + side // 2)
    diff = np.abs(np.clip(xy_min, None, 0))
    xy_min += diff
    xy_max += diff
    centroids[..., :2] -= diff
    diff = np.abs(np.clip(shape - xy_max, None, 0))
    xy_min -= diff
    xy_max -= diff
    centroids[..., :2] += diff
    xy_min, xy_max = xy_min.astype(np.int), xy_max.astype(np.int)
    crop = image[xy_min[0]: xy_max[0], xy_min[1]: xy_max[1]]
    centroids = np.array([ c for c in centroids if all(c[:2] >= 0) and all(c[:2] < side) ])
    return crop, centroids


def process_mask_idx(image, mask, sample, detector=None, otsu=False):
    mask = mask.copy()
    for c in np.unique(mask[mask!=0]):
        query = sample.query('id_segment==@c').kalcinaty.values
        roi = mask == c

        if not len(query):
            mask[roi] = 1
            continue

        query = query[0]
        mask[roi] = 1
        if detector is not None:
            thresholed_dots = detector.evaluate(
                image[np.newaxis].astype(np.float), roi[np.newaxis])
            roi = scipy.ndimage.binary_opening(detector.thresholed_dots[0], iterations=4)
        if otsu:
            xy_max = np.array(np.where(roi)).T.max(0)
            xy_min = np.array(np.where(roi)).T.min(0)

            crop = image[xy_min[0]: xy_max[0], xy_min[1]: xy_max[1]]
            if not crop.sum():
                continue
            val = filters.threshold_otsu(crop)
            roi[xy_min[0]: xy_max[0], xy_min[1]: xy_max[1]] *= (crop > val)
            roi = scipy.ndimage.binary_opening(roi, iterations=4)

        mask[roi] = kalcinaty_map[query] if query in kalcinaty_map.keys() else 1
    return mask


class SegmentCentroids:
    def __init__(self, detector: en.Pores, max_distance=150, plot=False):
        self.detector = detector
        self.plot = plot
        self.max_distance = max_distance

    def find_dots(self):
        thresholed_dots = self.detector.evaluate(self.crop[np.newaxis].astype(np.float))
        self.labeled, colours = scipy.ndimage.label(self.detector.thresholed_dots[0])
        self.olabeled = self.labeled.copy()

    def watershed_fn(self):
        self.markers = np.zeros_like(self.crop, dtype=np.int)
        for c in self.ctds:
            self.markers[c[0], c[1]] = c[2] # c -> (y, x, id_segment, points_nb, ..)
        self.mask = scipy.ndimage.distance_transform_bf(self.markers == 0)
        self.mask[self.mask > self.max_distance] = self.max_distance

        self.watershed = cv2.watershed(
            np.dstack([self.mask]*3).astype(np.uint8), 
            self.markers.astype(np.int32)
        )
        
    def init_figure(self):
        self.fig, self.ax = plt.subplots(2, 4, figsize=(20, 10))
        self.fig.tight_layout()
        [ el.axis('off') for a in self.ax for el in a ];

        self.ax[0][0].imshow(self.crop)
        for c in self.ctds:
            self.ax[0][0].scatter(c[1], c[0])
        self.ax[0][1].imshow(self.detector.dots[0])
        self.ax[0][2].imshow(self.mask)
        self.ax[0][3].imshow(self.watershed * (self.mask < self.max_distance))

        self.ax[1][0].imshow(self.olabeled > self.olabeled.min())

    def calculate_weights(self):
        self.weights = self.mask * (self.labeled > 0) + 1
        self.labeled[self.weights >= self.max_distance] = 0
        self.weights[self.weights >= self.max_distance] = 0

        for c in np.unique(self.labeled[self.labeled != 0]):
            roi = self.labeled == c
            val = self.weights[roi].min() - roi.sum() ** .5
            self.weights[roi] = val if val else 1e-3

        self.mask = np.zeros_like(self.crop, dtype=np.int)
        for m in np.unique(self.markers[self.markers != 0]):
            roi = self.watershed == m
            wroi = self.weights[roi]
            minw = wroi[wroi != 0].min()
            self.mask += (
                self.olabeled 
                == self.olabeled[self.weights == minw][0]
            ) * m
            
    def show(self):
        self.ax[1][1].imshow(self.mask * (self.labeled > 0))
        self.ax[1][2].imshow(self.weights)
        self.ax[1][3].imshow(self.mask)
        self.fig.show()
        plt.show()

    def __call__(self, crop, centroids):
        self.crop = crop.astype(np.float) + 1e-3
        self.ctds = centroids

        self.find_dots()
        self.watershed_fn()

        if self.plot:
            self.init_figure()

        self.calculate_weights()

        if self.plot:
            self.show()

        return self.mask
