import torch
from torch import nn

import numpy as np
from ..configs import config
from . import dataset as ds

import matplotlib.pyplot as plt
from tqdm import tqdm


def to_single_channel(model):
    try:
        return albunet_to_single_channel(model)
    except:
        pass
    try:
        return resnet_to_single_channel(model)
    except:
        raise('Error')


def resnet_to_single_channel(model):
    conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    nstd = { 
        key: param.sum(1).unsqueeze(1) 
        for key, param in model.conv1.state_dict().items()
    }
    print('Summed over: {}'.format(' | '.join(nstd.keys())))

    conv1.load_state_dict(nstd)
    model.conv1 = conv1
    return model


def albunet_to_single_channel(model):
    conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    nstd = { 
        key: param.sum(1).unsqueeze(1) 
        for key, param in model.conv1[0].state_dict().items()
    }
    print('Summed over: {}'.format(' | '.join(nstd.keys())))

    conv1.load_state_dict(nstd)
    model.conv1[0] = conv1
    model.encoder.conv1 = model.conv1[0]
    return model


def get_model(model, checkpoint=None, map_location=None, devices=None, drop_fc=False):
    model.cuda()

    if checkpoint is not None:
        sd = torch.load(checkpoint, map_location)#.module.state_dict()
        msd = model.state_dict()
        if drop_fc:
            msd = { 
                k: v for k, v in model.state_dict().items() 
                if k not in ['fc.weight', 'fc.bias']
            }
        sd = {k: v for k, v in sd.items() if k in msd}
        print('Overlapped keys: {}'.format(len(sd.keys())))
        msd.update(sd)
        model.load_state_dict(msd, strict=False)

    if devices is not None:
        model = torch.nn.DataParallel(model, device_ids=devices)

    return model


def freeze(model, unfreeze=False):
    children = list(model.children())
    if hasattr(model, 'children') and len(children):
        for child in children:
            freeze(child, unfreeze)
    elif hasattr(model, 'parameters'):
        for param in model.parameters():
            param.requires_grad = unfreeze


def unfreeze_bn(model):
    if isinstance(model, torch.nn.BatchNorm2d):
        for param in model.parameters():
            param.requires_grad = True

    children = list(model.children())
    if len(children):
        for child in children:
            unfreeze_bn(child)
    return None


# def get_target(data):
#     if mask in 


class Learner:
    def __init__(self, model, loss, opt, callbacks=[]):
        self.model = model
        self.loss = loss
        self.opt = opt
        if self.opt is not None:
            for group in self.opt.param_groups:
                group.setdefault('initial_lr', group['lr'])
        self.callbacks = callbacks

    def make_step(self, data, training=False):
        image = torch.autograd.Variable(data['image']).cuda()
        inference = config.TARGET_NAME not in data.keys()

        if not inference:
            target = torch.autograd.Variable(
                data[config.TARGET_NAME]).cuda()

        prediction = self.model(image).float()
        if not inference:
            losses = { 'loss': self.loss(prediction, target.float()) }
            target = target.data.cpu().numpy()

        prediction = torch.sigmoid(
            prediction,
        ).data.cpu().numpy()

        if training:
            losses['loss'].backward()
            self.opt.step()

        image = image.data.cpu().numpy()
        for k, loss in losses.items():
            losses[k] = loss.data.cpu().numpy()
        for callback in self.callbacks:
            losses.update(callbacks(
                prediction, target if inference else None, data))

        return losses

    def train_on_epoch(self, datagen, hard_negative_miner=None, lr_scheduler=None):
        self.model.train()
        torch.cuda.empty_cache()
        meters = list()

        for data in tqdm(datagen):
            self.opt.zero_grad()

            meters.append(self.make_step(data, training=True))
            if lr_scheduler is not None:
                if hasattr(lr_scheduler, 'batch_step'):
                    lr_scheduler.batch_step(logs=meters[-1])

            if hard_negative_miner is not None:
                hard_negative_miner.update_cache(meters[-1], data)
                if hard_negative_miner.need_iter():
                    self.make_step(hard_negative_miner.get_cache(), training=True)
                    hard_negative_miner.invalidate_cache()

        self.opt.zero_grad()
        torch.cuda.empty_cache()
        return meters # metrics.aggregate(meters)

    def validate(self, datagen):
        self.model.eval()
        meters = list()

        with torch.no_grad():
            for data in tqdm(datagen):
                meters.append(self.make_step(data, training=False))

        return meters # metrics.aggregate(meters)

    def inference(self, datagen, callbacks=[]):
        self.model.eval()
        self.callbacks = callbacks

        with torch.no_grad():
            for data in tqdm(datagen):
                meters.append(self.make_step(data, training=False))

        return meters # metrics.aggregate(meters)        

    def infer_on_data(self, data, verbose=True):
        if self.model.training:
            self.model.eval()
        if len(data['image'].shape) == 3:
            data['image'] = data['image'].unsqueeze(0)

        image = torch.autograd.Variable(data['image']).cuda()
        pred = self.model.forward(image)
        pred = torch.sigmoid(pred).data.cpu().numpy()
        if verbose:
            image = np.rollaxis(data['image'][0].numpy(), 0, 3)
            image = (
                image * np.array(config.STD) + np.array(config.MEAN))

            fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
            ax[0][0].imshow(np.squeeze(image))
            if not isinstance(data['mask'], np.ndarray):
                data['mask'] = data['mask'].data.numpy()
            ax[0][1].imshow(np.squeeze(data['mask'])[0])
            cs = ax[0][2].imshow(np.squeeze(pred[0])[0])
            if not isinstance(data['mask'], np.ndarray):
                data['mask'] = data['mask'].data.numpy()
            ax[1][1].imshow(np.squeeze(data['mask'])[1])
            cs = ax[1][2].imshow(np.squeeze(pred[0])[1])
            fig.colorbar(cs)
            plt.show()

        return pred

    def save(self, path):
        state_dict = self.model.state_dict()
        if isinstance(self.model, torch.nn.DataParallel):
            state_dict = self.model.module.state_dict()
        torch.save(state_dict, path)
        print('Saved in {}:'.format(path))

    def freeze_encoder(self, unfreeze=False):
        if hasattr(self.model, 'module'):
            encoder = self.model.module.encoder
        elif hasattr(self.model, 'encoder'):
            encoder = self.model.encoder
        thf.freeze(encoder, unfreeze=unfreeze)
        thf.unfreeze_bn(encoder)
