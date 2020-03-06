from albumentations import (
    CLAHE, RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion, 
    GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, 
    MedianBlur, IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast,
    RandomBrightness, Flip, OneOf, Compose, ToGray, InvertImg, HorizontalFlip,
    RandomCrop, CenterCrop
)

import cv2

from ..configs import config


class Augmentation:
    def __init__(self, strength=1., key_points=False):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        self.coeff = int(3 * strength)
        self.k = max(1, self.coeff if self.coeff % 2 else self.coeff - 1)
        self.strength = strength
        self.params = {}
        if key_points:
            self.params = { 'keypoint_params': {'format': 'yx'} }
        self.augs = self.get_augmentations()

    def __call__(self, data):
        if self.augs is not None:
            data = self.augs(**data)
        return data

    def get_photometric(self):
        return Compose([
            OneOf([
                CLAHE(clip_limit=2, p=.8),
                IAASharpen(p=.8),
                IAAEmboss(p=.8),
            ], p=0.6),
            OneOf([
                IAAAdditiveGaussianNoise(p=.6),
                GaussNoise(p=.7),
            ], p=.5),
            OneOf([
                MotionBlur(p=.5),
                MedianBlur(blur_limit=self.k, p=.3),
                Blur(blur_limit=self.k, p=.5),
            ], p=.5),
            OneOf([
                RandomContrast(),
                RandomBrightness(),
            ], p=.8),
        ], p=0.95)

    def get_geoometric(self):
        geometric = [
            ShiftScaleRotate(
                shift_limit=0.0625, 
                scale_limit=(-.1, .4), 
                rotate_limit=45, 
                border_mode=0,
                p=.95
            ),
        ]
        return Compose(geometric)

    def get_augmentations(self):
        if self.strength is None:
            return None

        transformations = [
            Compose([
                Flip(),
                RandomRotate90(),
            ], p=1.),
            Compose([
                self.get_photometric(),
                self.get_geoometric(),
            ], p=.95),
            RandomCrop(config.CNN_SIDE, config.CNN_SIDE, always_apply=True)
        ]
        return Compose(
            transformations,
            **self.params
        )


class ValidAugmentation(Augmentation):

    def get_augmentations(self):
        if self.strength is None:
            return None

        transformations = [
            Compose([
                CenterCrop(config.CROP_SIDE, config.CROP_SIDE, always_apply=True),
            ], p=1.),
        ]
        return Compose(
            transformations
        )


class TestAugmentation(Augmentation):

    def get_augmentations(self):
        if self.strength is None:
            return None

        transformations = [
            Compose([
#                 HorizontalFlip(),
            ], p=1.),
        ]
        return Compose(
            transformations,
            keypoint_params={'format': 'yx'}
        )
