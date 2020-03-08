import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import cv2
import os

from ..configs import config
from ..modules import dataset as ds
# from ..modules import metrics as ms


LABEL_COLOUR = {
    2: (1., 1., .6),
    1: (1., 1., 1.),
    0: (1., .6, .6),
    -1: (1., 1., 1.)
}


def plot_discrepancy(discrepancy, labels=config.ANNOTATORS_ORDER, save_path=None, mask=True):
    # Generate a mask for the upper triangle
    if mask:
        mask = np.triu(np.ones_like(discrepancy, dtype=np.bool), k=1)
    else:
        mask = None

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(9, 7))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    fmt = 'd' if discrepancy.dtype.char in np.typecodes['Integer'] else '.2g'
    # Draw the heatmap with the mask and correct aspect ratio
    chart = sns.heatmap(discrepancy, mask=mask, cmap=cmap, fmt=fmt,
                xticklabels=labels, yticklabels=labels,
                square=True, linewidths=.5, cbar_kws={"shrink": 1.}, annot=True)
    chart.set_yticklabels(chart.get_yticklabels(), rotation=-50)

    if save_path is not None:
        f.savefig(save_path)


def visualize_all_sides(sides: dict):
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
    fig.tight_layout()
    keys = sorted(list(sides.keys()))
    axes[0][0].imshow(np.dstack([sides[keys[0]]] * 3))
    axes[0][0].axis('off')
    axes[0][1].imshow(np.dstack([sides[keys[1]]] * 3))
    axes[0][1].axis('off')
    axes[1][0].imshow(np.dstack([sides[keys[2]]] * 3))
    axes[1][0].axis('off')
    axes[1][1].imshow(np.dstack([sides[keys[3]]] * 3))
    axes[1][1].axis('off')
    plt.show()


def make_image_row(images, subax, titles=None):
    for i, image in enumerate(images):
        subax[i].imshow(np.dstack([image] * 3))
        if titles:
            subax[0].set_title(titles[i])
        subax[i].axis('off')
    return subax


def plot_pull(keys, names, titles=None, save_name=None):
    fig, ax = plt.subplots(max(2, len(names)), 4, figsize=(20, max(2, len(names)) * 5))
    fig.tight_layout()

    for i, name in enumerate(names):
        images = list()
        for k in sorted(list(name.keys())):
            v = name[k]
            path = '{}.'.join((str(config.PATHS.PNG/keys[i]/v), 'png'))
            image = cv2.imread(path.format(''), 0)
            mask = cv2.imread(path.format('_class'), 0)

            if mask is not None:
                roi = mask > 0
                image[roi] = np.min([
                    (image[roi].astype(np.int) + 100) * 1.2, 
                    np.array([255] * roi.sum())
                ], axis=0).astype(np.uint8)
            images.append(image)
        make_image_row(
            images, ax[i], 
            None if titles is None else [titles[i]] * len(images)
        )

    if save_name:
        plt.savefig(os.path.join(config.PATHS.LOGS, 'patients_{}.png'.format(save_name)))
    plt.show()
    

def vis_points(image, points, diameter=5):
    im = image.copy()

    for (x, y) in points:
        cv2.circle(im, (int(x), int(y)), diameter, (5, 255, 0), -1)

    return im


def visualize_bbox(img, bbox, category_id_to_name, colour=LABEL_COLOUR, thickness=2):
    try:
        bbox = bbox.data.numpy()
    except:
        pass
    x_min, y_min, x_max, y_max, class_id = bbox.astype(np.int)
    
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(1.,), thickness=thickness)
    return img


def visualize_bboxes(pred, annotations, category_id_to_name):
    _, axes = plt.subplots(ncols=2, figsize=(15, 7))
    img = annotations['image'][..., 0]
    if not isinstance(img, np.ndarray):
        img = annotations['image'].data.numpy().copy()[0]
    if img.dtype != np.uint8:
        img = (img * config.STD) + config.MEAN
    if len(img.shape) == 2:
        img = np.dstack([img] * 3)
    img1, img2 = img.copy(), img.copy()
    for idx, bbox in enumerate(pred['bboxes']):
        img1 = visualize_bbox(img1, bbox, category_id_to_name)
    for idx, bbox in enumerate(annotations['bboxes']):
        img2 = visualize_bbox(img2, bbox, category_id_to_name)
    axes[0].imshow(img1)
    axes[1].imshow(img2)
    plt.show()


def plot_losses(history):
    _, axes = plt.subplots(ncols=2, figsize=(20, 6))
    axes[0].plot([l['loss'] for l in history['train']], label='weighted loss train', alpha=0.8)
    axes[0].plot([l['reg_loss'] for l in history['train']], label='reg loss train', alpha=0.5)
    axes[0].plot([l['clf_loss'] for l in history['train']], label='clf loss train', alpha=0.5)

    axes[0].plot([l['loss'] for l in history['valid']], label='weighted loss val', alpha=0.8)
    axes[0].plot([l['reg_loss'] for l in history['valid']], label='reg loss val', alpha=0.5)
    axes[0].plot([l['clf_loss'] for l in history['valid']], label='clf loss val', alpha=0.5)

    axes[0].set_title('Losses')
    axes[0].legend()
    axes[0].grid()

    axes[1].plot([l['map_all'] for l in history['valid']], label='mAP all val', alpha=0.7)
    axes[1].plot([l['map_pathology'] for l in history['valid']], label='mAP pathology val', alpha=0.7)

    axes[1].plot([l['map_all'] for l in history['train']], label='mAP all train', alpha=0.7)
    axes[1].plot([l['map_pathology'] for l in history['train']], label='mAP pathology train', alpha=0.7)

    axes[1].set_title('Meterics')
    axes[1].legend()
    axes[1].grid()

    plt.show()


def visualise_pred(data):
    annotation, bboxes, scores = ms._extract_meta(data)
    filename = data['pid']
    image = cv2.imread(os.path.join(config.PATHS.IMAGES, filename))
    image = ds.img_transform(image)

    fdata = {
        'image': image,
        'bboxes': bboxes
    }
    tdata = {
        'image': image,
        'bboxes': annotation
    }
    visualize_bboxes(tdata, fdata, {0:'', 1:''})