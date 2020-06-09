from skimage import draw
import numpy as np
import re
import os
import cv2
import scipy
import scipy.stats
import sklearn.mixture
import json
from PIL import Image
import pandas as pd
import easydict
from collections import defaultdict
from glob import glob
import pydicom as dicom

from ..configs import config


columns = [
    'case', 'which_side', 'dcm_filename', 'id_segment', 
    'path', 'pixel_spacing_y', 'pixel_spacing_x', 'rows', 'columns',
    'relative_xray_exposure', 'sensitivity', 'area', 'shape', 'coeff',
    'centroid_y', 'centroid_x', 'size_y', 'size_x', 'points_nb'
]


def convert_and_unify(troot, file):
    try:
        src = os.path.join(troot, file)
        dcm = dicom.read_file(src)
        instance_n = int(dcm.InstanceNumber) % 10
        dst = os.path.join(troot, 'a%05d.png' % instance_n)
        
        image = dcm.pixel_array.copy()

        median = np.median(image)
        c1 = (image == median).sum() / (image == median + 1).sum() ** .5
        c2 = (image == median).sum() / (image == median - 1).sum() ** .5

        # if min(c1, c2) > 10:
        try:
            GM = sklearn.mixture.GaussianMixture(
                n_components=2, 
                covariance_type='spherical')
            GM.fit(image.flatten().reshape((-1, 1)))
            image_mean = GM.means_[np.argmax(GM.covariances_)]
            back_mean = GM.means_[np.argmin(GM.covariances_)]
            image_std = np.sqrt(GM.covariances_).max()
        except:
            GM = sklearn.mixture.GaussianMixture(
            n_components=1, 
            covariance_type='spherical')
            GM.fit(image[image != median].reshape((-1, 1)))
            image_mean = GM.means_[np.argmax(GM.covariances_)]
            image_std = np.sqrt(GM.covariances_).max()
            back_mean = median

        mid = 2 ** 8 if image.dtype == np.uint8 else 2 ** 16
        if back_mean > image_mean:
            image = mid - 1 - image
            image_mean = mid - 1 - image_mean

        if image.dtype != np.uint8:
            image = np.clip(
                image, 
                image_mean - image_std * 1, 
                image_mean + image_std * 5)
            image = (image - image.min()) / (image.max() - image.min())
            image = (image * 255).astype(np.uint8)

        cv2.imwrite(dst, image)
        return { src: dst }
    except dicom.errors.InvalidDicomError:
        return None


def draw_circle(y, x, shape):
    mask = np.zeros(shape=shape, dtype=np.bool)
    coords = draw.circle(y[0], x[0], config.NIPPLE_RADIUS, shape=shape)
    mask[coords] = True
    return mask


def build_path(path, root, template='case_{}-side_{}.png', replace=True):
    fname = os.path.basename(path)
    side = int(re.findall(config.NUM_REGEX, fname)[0])
    case = os.path.basename(os.path.dirname(path))
    path = os.path.join(root, template.format(case, side))
    if replace:
        path = path.replace('.png', '_{}.png')
    return path


def load_row_images(row):
    case = row['Кейс']
    path = (case.split('_')[0], case)
    path = os.path.join(config.PATHS.DATA_ROOT, *path, '{}')

    path_map = glob(path.format('*.dcm'))
    path_map = { int(dicom.read_file(p).InstanceNumber): p for p in path_map }
    
    sides = { 
        k: dicom.read_file(v).pixel_array
        for k, v in path_map.items()
    }

    keys = sorted(list(sides.keys()))
    sides = { 
        i: sides[k] if i % 2 else sides[k][:, ::-1] 
        for i, k in enumerate(keys) 
    }

    return sides


def crop_sides(sides):
    cropped = {}

    for i in range(2):
        xy_min = list()
        xy_max = list()
        for k in np.arange(i * 2, i * 2 + 2):
            mr = scipy.stats.mode(sides[k].flatten())
            roi = scipy.ndimage.binary_opening(sides[k] > (mr.mode + 4), iterations=5)
            labeled, _ = scipy.ndimage.label(roi)
            roi = labeled == np.argmax(np.bincount(labeled[labeled != 0]))

            coords = np.array(np.where(roi)).T
            xy_min.append(coords.min(0))
            xy_max.append(coords.max(0))

        xy_min = np.array(xy_min).min(axis=0)
        xy_max = np.array(xy_max).max(axis=0)

        for k in np.arange(i * 2, i * 2 + 2):
            cropped[k] = sides[k][
                xy_min[0]: xy_max[0], 
                xy_min[1]: xy_max[1]
            ]
    return cropped


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def crop_id(path): 
    return int(
        re.findall(
            config.NUM_REGEX, 
            path
        )[0]
    )


def to_str(x):
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return ' '.join(map(str, x))
    return x


def protocol_general(row, case):
    path = config.PATHS.CROPS/case
    shape = Image.open(path/row.filename).size[::-1]
    coeff = max(shape) / max(row.imageHeight, row.imageWidth)
    
    row = easydict.EasyDict(row)
    upd = { 
        'id_segment': row.id + 1, # 0 -> 1 for mask colours
        'case': case,
        'shape': shape,
        'area': 0,
        'coeff': coeff,
        'path': path,
        'which_side': crop_id(row.filename),
    }
    row.update(upd)
    return row


def protocol_mammography(row, case):
    row = protocol_general(row, case)

    coords = np.array([ [p['y'], p['x']] for p in row.points ])
    if all(coords[-1] == coords[0]) and len(coords) > 1:
        coords = coords[:-1]

    coords = row.coeff * np.array(coords)
    size_y, size_x = coords.max(axis=0) - coords.min(axis=0)
    y, x = coords.mean(axis=0)

    row.update({ 
        'centroid_y': y, 
        'centroid_x': x, 
        'size_y': size_y,
        'size_x': size_x,
        'points_nb': len(row.points) 
    })

    row = pd.Series(row)

    to_drop = set(row.keys()).difference(columns)
    return row.drop(to_drop)


def extract_birads_old(lines):
    regexp = re.compile(r"(?s)(^|\n)\s*.*(?P<BIRADS>(birads)[^\n]*)", re.IGNORECASE)
    reiter = re.finditer(regexp, lines)
    return [ 
        m.group('BIRADS') for m in reiter 
    ]


def to_int(v):
    maps = {
        'з': 3,
    }
    try: return int(v)
    except: return int(maps[v])


def extract_birads(lines):
    regexp = r"(?s)(^|\n)\s*.*(?P<LINE>(birads)[^\n]*{}[^\n]*)"
    regexp = regexp.format(r"кат[а-я]*(|\s)*(?P<BIRADS>[0-6з])")
    regexp = re.compile(regexp, re.IGNORECASE)
    reiter = re.finditer(regexp, lines)
    groups = [ 
        to_int(m.group('BIRADS')) for m in reiter 
    ]
    return groups.pop() if groups else None


def dicom2png(case):
    path = (case.split('_')[0], case)
    path = os.path.join(config.PATHS.DATA_ROOT, *path, '{}')

    path_map = glob(path.format('*.dcm'))
    path_map = { int(dicom.read_file(p).InstanceNumber): p for p in path_map }

    for k, v in path_map.items():
        path = config.PATHS.CROPS/case
        os.makedirs(str(path), exist_ok=True)
        name = os.path.basename(v.replace('.dcm', '.png'))
        regexp = re.compile(r"^(?P<prefix>[a-zA-Z]+)([0-9]+)(?P<format>\.[a-z]{0,5})$")
        path /= re.sub(regexp, r"\g<prefix>%05d\g<format>" % k, name)
        dcm = dicom.read_file(v).pixel_array
        if config.CROP_SIDE is not None:
            shape = np.array(dcm.shape)[:2]
            shape_ = (shape * (config.CROP_SIDE / shape.min())).astype(np.int)
            dcm = cv2.resize(dcm, tuple(shape_[::-1].tolist()))
        cv2.imwrite(str(path), dcm)

    return { case: dcm.shape }


def extract_masks(el, row):
    x = np.array([ v['x'] for v in el.points ])
    y = np.array([ v['y'] for v in el.points ])

    extractor = poly2mask
    if el.fieldname == 'centroid-grudnogo-soska':
        extractor = draw_circle

    mask = extractor(
        y * row.coeff, x * row.coeff, (row['shape'][0], row['shape'][1]))
    row.area = mask.sum()
    return mask.astype(np.int)


def resize_image(image, interpolation=2, side=config.MIN_SIDE):
    if side is None:
        return image
    if image is None:
        return None
    shape = np.array(image.shape)[:2]
    shape_ = (shape * (side / shape.min())).astype(np.int)
    return cv2.resize(
        image, tuple(shape_[::-1].tolist()), interpolation=interpolation)


def process_annotations_RoI(annot):
    data = pd.DataFrame(columns=columns)
    xml = decode_json(annot.XML)
    xml += decode_json(annot.XML_x)

    if not xml: return None

    xmls = defaultdict(list)
    for x in xml:
        if 'fieldname' not in x.keys(): return None
        xmls[x['fieldname']].append(x)

    for fieldname, xml in xmls.items():
        for el in xml:
            el = easydict.EasyDict(el)
            row = protocol_mammography(el, annot['Кейс'])
            mask = extract_masks(el, row)
            row = pd.concat([row, annot[config.ANNOTATION_COLUMNS]])
            row.fieldname = fieldname
            data = data.append(row, ignore_index=True)
            path = str(os.path.join(row.path, el.filename)).replace('.png', '_{}.png')
            cv2.imwrite(path.format(fieldname), mask)

    return data


jd = json.JSONDecoder()
def decode_json(json):
    xml = jd.decode(json)
    if not xml:
        return xml

    if isinstance(xml, dict):
        xml = list(xml.values())
    return xml
