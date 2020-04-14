from albumentations import (
    CLAHE, RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion, 
    GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, 
    MedianBlur, IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast,
    RandomBrightness, Flip, OneOf, Compose, ToGray, InvertImg, HorizontalFlip,
    RandomCrop, CenterCrop
)

import cv2
import numpy as np

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

def _rotate_mirror_do(im):
    """
    Duplicate an np array (image) of shape (x, y, nb_channels) 8 times, in order
    to have all the possible rotations and mirrors of that image that fits the
    possible 90 degrees rotations.
    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    mirrs = []
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(-2, -1), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(-2, -1), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(-2, -1), k=3))
    im = np.array(im)[..., ::-1]
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(-2, -1), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(-2, -1), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(-2, -1), k=3))
    return mirrs


def _rotate_mirror_undo(im_mirrs):
    """
    merges a list of 8 np arrays (images) of shape (x, y, nb_channels) generated
    from the `_rotate_mirror_do` function. Each images might have changed and
    merging them implies to rotated them back in order and average things out.
    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    origs = []
    origs.append(np.array(im_mirrs[0]))
    origs.append(np.rot90(np.array(im_mirrs[1]), axes=(-2, -1), k=3))
    origs.append(np.rot90(np.array(im_mirrs[2]), axes=(-2, -1), k=2))
    origs.append(np.rot90(np.array(im_mirrs[3]), axes=(-2, -1), k=1))
    origs.append(np.array(im_mirrs[4])[..., ::-1])
    origs.append(np.rot90(np.array(im_mirrs[5]), axes=(-2, -1), k=3)[:, :, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[6]), axes=(-2, -1), k=2)[:, :, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[7]), axes=(-2, -1), k=1)[:, :, ::-1])
    return np.stack(origs).prod(axis=0) ** (1 / len(origs))
