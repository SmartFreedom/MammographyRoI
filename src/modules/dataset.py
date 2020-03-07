import os
import cv2
import numpy as np
import pandas as pd
import sklearn.model_selection
import imageio
import easydict
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Normalize, Compose

from ..configs import config
from ..modules import augmentations as augs


img_transform = Compose([
    ToTensor(),
    Normalize(mean=config.MEAN, std=config.STD)
])


def collater(data):
    collated = easydict.EasyDict()
    collated.image = torch.stack([
        s['image'] for s in data
    ], 0) # here (0::2 + 1::2) / 2
    collated.mask = torch.FloatTensor([ 
        s['mask'] for s in data 
    ]).unsqueeze(1)
    collated.pid = [s['pid'] for s in data]

    return collated


def stack_aug_collater(data):
    collated = easydict.EasyDict()
    
    collated.image = torch.stack([ 
        (a['image'] + b['image']) / 2. 
        for a, b in zip(data[::2], data[1::2]) 
    ])
    collated.mask = torch.FloatTensor(
        np.array([
            np.max([a['mask'], b['mask']], axis=0)
            for a, b in zip(data[::2], data[1::2])
        ])
    ).unsqueeze(1)
    collated.pid = [s['pid'] for s in data]

    return collated


class SegmentationDataset(Dataset):
    def __init__(self, data: pd.DataFrame, root=config.PATHS.EXPERIMENT_DATA, augmentations=None):
        self.augmentations = augmentations
        self.data = data.copy()
        self.groups = data.groupby('fileid').groups
        self.keys = list(self.groups.keys())
        self.root = (root)

    def __getitem__(self, idx):
        fileid = self.keys[idx]
        data = self.data.loc[self.groups[fileid]]
        image = self.load_image(fileid)
        mask = self.load_annotations(fileid)

        data = {
            "image": image, 
            "mask": mask, 
            "pid": fileid,
        }

        if self.augmentations is not None:
            data = self.augmentations(data)

        return self.postprocess(data)

    def postprocess(self, data):
        data = easydict.EasyDict(data)
        # image correction
        pad = 32 - data.image.shape[1] % 32 if data.image.shape[1] % 32 else 0
        image = np.pad(data.image, (0, pad), mode='constant')
        data.mask = np.pad(data.mask, (0, pad), mode='constant')
        data.image = img_transform(np.expand_dims(image, -1))

        return data

    def load_image(self, fileid):
        return cv2.imread(os.path.join(self.root, fileid), 0)

    def load_annotations(self, fileid):
        fileid = fileid.split('.')
        fileid[-2] = fileid[-2] + '_RoI_mask'
        fileid = '.'.join(fileid)
        return cv2.imread(os.path.join(self.root, fileid), 0)

    def num_classes(self):
        return 2

    def __len__(self):
        return len(self.keys)


class ExtendedSampler(Sampler):
    r"""Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        idxs = np.arange(len(self.dataset)).tolist() * config.DATASET_LEN_MULTIPLIER
        np.random.shuffle(idxs)
        return iter(idxs)

    def __len__(self):
        return len(self.dataset) * config.DATASET_LEN_MULTIPLIER


def get_datagens(train_df, valid_df, dataset_type=SegmentationDataset, 
                 augmentations=augs.Augmentation(),
                 valid_augmentations=augs.ValidAugmentation(),
                 collaters=None, sampler=None, **kwargs):
    
    dataset_train = dataset_type(
        data=train_df,
        augmentations=augmentations,
        **kwargs
    )
    dataset_valid = dataset_type(
        data=valid_df,
        augmentations=valid_augmentations,
        **kwargs
    )

    kwargs = {} if collaters is None else { 'collate_fn': collaters['train'] }
    train_datagen = torch.utils.data.DataLoader(
        dataset_train, pin_memory=False, 
        shuffle=True if sampler is None else None, 
        sampler=sampler(dataset_train) if sampler is not None else None, 
        batch_size=config.BATCH_SIZE, num_workers=config.WORKERS_NB,
        **kwargs
    )
    kwargs = {} if collaters is None else { 'collate_fn': collaters['valid'] }
    valid_datagen = torch.utils.data.DataLoader(
        dataset_valid, batch_size=config.BATCH_SIZE_TEST, 
        num_workers=config.WORKERS_NB, 
        **kwargs
    )
    return train_datagen, valid_datagen
