from skimage import draw
import numpy as np
import re
import os
import cv2
import scipy
import json
from PIL import Image
import pandas as pd
import easydict
from glob import glob
import pydicom as dicom

from ..configs import config


columns = [
    'case', 'which_side', 'dcm_filename',
    'id_segment', 'kalcinaty', 'lokalnaya-perestrojka-struktury-mzh', 
    'vneochagovie-kalcinaty-2', 'klass-po-bi-rads',
    'obrazovaniya', 'inorodnye-tela-4', 'path',
    'pixel_spacing_y', 'pixel_spacing_x', 'rows', 'columns',
    'relative_xray_exposure', 'sensitivity', 'area', 'shape', 'coeff',
    'centroid_y', 'centroid_x', 'size_y', 'size_x', 'points_nb'
]


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
    path = config.PATHS.PNG/case
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
    upd_keys = [
        'kalcinaty', 'lokalnaya-perestrojka-struktury-mzh',
        'obrazovaniya', 'inorodnye-tela-4', 
        'vneochagovie-kalcinaty-2', 'klass-po-bi-rads'
    ]
    # it's happends that formData might be missed in annotation
    upd = { k: -2 for k in upd_keys }
    if 'formData' in row.keys():
        upd = { 
            k: (
                to_str(row.formData[k]) 
                if row.formData[k] 
                else '-1'
            )
            if k in row.formData.keys() 
            else '-1' for k in upd_keys 
        }
    row.update(upd)

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
        path = config.PATHS.PNG/case
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


class_map = {
    False: 128, # row.kalcinaty > 1
    True: 255, # for those which aren't kalcinaty
}
def extract_masks(el, row, masks):
    x = np.array([ v['x'] for v in el.points ])
    y = np.array([ v['y'] for v in el.points ])

    mask = poly2mask(y * row.coeff, x * row.coeff, (row['shape'][0], row['shape'][1]))
    idx_mask = (el.id + 1) * mask.astype(np.int)
    class_mask = class_map[
        (int(row.kalcinaty) > 1) or (int(row['vneochagovie-kalcinaty-2']) > 0)
    ] * mask.astype(np.int)

    masks[el.filename] = {
        'idx': np.max([idx_mask, masks[el.filename]['idx']], axis=0) 
            if el.filename in masks.keys() 
            else idx_mask,
        'class': np.max([class_mask, masks[el.filename]['class']], axis=0) 
            if el.filename in masks.keys() 
            else class_mask,
    }
    row.area = mask.sum()


def process_annotations_RoI(annot):
    data = pd.DataFrame(columns=columns)
    xml = decode_json(annot.XML)

    if not xml: return None

    masks = {}
    for el in xml:
        el = easydict.EasyDict(el)

        row = protocol_mammography(el, annot['Кейс'])
        extract_masks(el, row, masks)
        row = pd.concat([row, annot[config.ANNOTATION_COLUMNS]])
        data = data.append(row, ignore_index=True)

    for k in masks.keys():
        path_ = str(os.path.join(row.path, k)).replace('.png', '_{}.png')
        masks_ = { mtn: cv2.imread(path_.format(mtn), 0) for mtn in masks[k].keys()}
        cv2.imwrite(path_.format('RoI_mask'), masks[k]['class'])

    return data



jd = json.JSONDecoder()
def decode_json(json):
    xml = jd.decode(json)
    if not xml:
        return xml

    if isinstance(xml, dict):
        xml = list(xml.values())
    return xml
