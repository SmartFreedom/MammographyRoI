import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


eps = 1e-3


def dice_round(preds, trues, is_average=True):
    preds = torch.round(preds)
    return dice_loss(preds, trues, is_average=is_average)


def multi_class_dice_round(preds, trues, is_average=True):
    preds = torch.round(preds)
    return multi_class_dice(preds, trues, is_average=is_average)


def dice_loss(preds, trues, weight=None, is_average=True):
    preds = preds.contiguous()
    trues = trues.contiguous()
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    if weight is not None:
        w = torch.autograd.Variable(weight).view(num, -1)
        preds = preds * w
        trues = trues * w
    intersection = (preds * trues).sum(1)
    scores = (2. * intersection + eps) / (preds.sum(1) + trues.sum(1) + eps)

    if is_average:
        score = scores.sum()/num
        return torch.clamp(score, 0., 1.)
    else:
        return scores


def per_class_dice(preds, trues, weight=None, is_average=True):
    loss = []
    for idx in range(1, preds.shape[1]):
        loss.append(dice_loss(preds[:,idx,...].contiguous(), (trues==idx).float().contiguous(), weight, is_average))
    return loss


def multi_class_dice(preds, trues, weight=None, is_average=True):
    channels = per_class_dice(preds, trues, weight, is_average)
    return sum(channels) / len(channels)


def jaccard_round(preds, trues, is_average=True):
    preds = torch.round(preds)
    return jaccard(preds, trues, is_average=is_average)


def jaccard(preds, trues, weight=None, is_average=True):
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    if weight is not None:
        w = torch.autograd.Variable(weight).view(num, -1)
        preds = preds * w
        trues = trues * w
    intersection = (preds * trues).sum(1)
    scores = (intersection + eps) / ((preds + trues).sum(1) - intersection + eps)

    score = scores.sum()
    if is_average:
        score /= num
    return torch.clamp(score, 0., 1.)


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        return dice_loss(input, target, self.weight, self.size_average)


class JaccardLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        return jaccard(input, target, self.weight, self.size_average)


class BCEDiceJaccardLoss(nn.Module):
    def __init__(self, weights, weight=None, size_average=True):
        super().__init__()
        self.weights = weights
        self.bce = nn.BCEWithLogitsLoss()
        self.jacc = JaccardLoss()
        self.dice = DiceLoss()
        self.mapping = {'bce': self.bce,
                        'jacc': self.jacc,
                        'dice': self.dice}
        self.values = {}

    def forward(self, input, target):
        loss = 0
        sigmoid_input = torch.sigmoid(input)
        for k, v in self.weights.items():
            if not v:
                continue

            val = self.mapping[k](
                input if k == 'bce' else sigmoid_input, 
                target
            )
            self.values[k] = val
            if k != 'bce':
                loss += self.weights[k] * (1 - val)
            else:
                loss += self.weights[k] * val
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

        if isinstance(alpha, (float, int)): 
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list): 
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, pred, target):
        if pred.dim() > 2:
            pred = pred.view(pred.size(0), pred.size(1), -1)  # N,C,H,W => N,C,H*W
            pred = pred.transpose(1, 2)    # N,C,H*W => N,H*W,C
            pred = pred.contiguous().view(-1, pred.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(pred)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != pred.data.type():
                self.alpha = self.alpha.type_as(pred.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()


class ConditionalBCE(nn.Module):
    def __init__(self):
        super(ConditionalBCE, self).__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        pred = pred.flatten()
        target = target.flatten()

        mask = (target != 1)
        idxs = torch.nonzero(mask).view(-1)
        target = (target.gather(0, idxs) > 0).float()
        pred = pred.gather(0, idxs).float()

        return self.loss(pred, target)


class WhetherCentroidPresentedBCE(nn.Module):
    def __init__(self):
        super(WhetherCentroidPresentedBCE, self).__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        mask = target[:, -1].reshape(shape=(target.size(0), -1)).sum(1)
        centroid_loss = 0
        if mask.sum():
            idxs = torch.nonzero(mask).view(-1)
            _target = (
                target[idxs][:, -1:] 
                - target[idxs][:, -1:] * target[idxs][:, :1])
            _pred = pred[idxs][:, -1:]
            centroid_loss = self.loss(_pred, _target)

        tissue_loss = self.loss(pred[:, :1], target[:, :1])
        target = target.reshape(shape=(target.size(0), target.size(1), -1))
        roi = 1 - (
            target[:, -1] - target[:, -1] * target[:, 0]
        ) * (1 - target[:, 1])
        idxs = torch.nonzero(roi.reshape(-1)).view(-1)
        target = target[:, 1].reshape(-1)[idxs]
        pred = pred[:, 1].reshape(-1)[idxs]

        whole_loss = self.loss(pred, target)
        return whole_loss + centroid_loss + tissue_loss
