from pathlib import Path
import os
import easydict
import pickle
import pandas as pd


MEAN = [74.77618355029848]
STD = [31.738553261533994]

BATCH_SIZE = 42
BATCH_SIZE_TEST = 1 #BATCH_SIZE // 3
CROP_SIDE = 512
CNN_SIDE = 512
BBOXES_SIDE = 50
DEVICES = [0, 1, 2]
WORKERS_NB = 16
CUDA_IDX = 0
CROP_STEP = 256
SIDE = 512
DROPOUT = .2
NIPPLE_RADIUS = 25

RANDOM_SEED = 42
DATASET_LEN_MULTIPLIER = 8
TRAIN_TEST_SPLIT = .9


OPT = easydict.EasyDict()
OPT.MULTIPLIER = .1
OPT.COEFF = 1.
OPT.INIT_LR = 1e-4
OPT.EPOCHS_PER_SCHEDULER = 25

TARGET_NAME = 'mask'


PATHS = easydict.EasyDict()
PATHS.DATA_ROOT = Path('/data/datasets/nas/20190715/Medicina')
PATHS.DATA_LOCAL = Path('/data/datasets/ssd/mammography')

PATHS.PNG = PATHS.DATA_LOCAL/'png'
PATHS.LOGS = PATHS.DATA_LOCAL/'roi_segmentation'/'logs'
PATHS.CSV = PATHS.DATA_LOCAL/'csv'
PATHS.CROPS = PATHS.DATA_LOCAL/'roi_segmentation'/'crops'
PATHS.EXPERIMENT_DATA = PATHS.DATA_LOCAL/'roi_segmentation'

ANNOTATIONS = [
        pd.read_csv(PATHS.CSV/'export_last.csv'),
]
ANNOTATIONS = pd.concat(ANNOTATIONS, sort=False).reset_index(drop=True)
ANNOTATORS_ORDER = pickle.load(open(PATHS.CSV/'annotators_order.pkl', 'rb'))


API = easydict.EasyDict()
API.ROOT = 'https://label.cmai.tech'
API.CASES = API.ROOT + '/api/v1/cases'
API.KEY = 'jMTCJiJNETMDpwystkl25dFgPbDVpmiSl0Cx6k5pZ7xcUNKu4hbLOpo2UWgIOq8ZBZ7U5Q1djTsyPdmoekNAU3RqhP2kMhp8A5Ef80YDLIchZOGNi77rUrsdlTatwEva'


NUM_REGEX = r"^.*?(?P<NUMS>[0-9]+)\.[a-zA-Z]+"


ASSYMETRY_TYPES = [
    "Нет",
    "Есть Локальная (очаговая)",
    "Есть Тотальная",
    "Есть Локальная с кальцинатами",
    "Есть Динамическая"
]


DENSITY_TYPES = [
    'A',
    'B',
    'C',
    'D'
]


ANNOTATION_COLUMNS = [
    'ID', 'Исследователь', 'ID Исследования', 'Кейс', 'Источник',
    'ID Группы', 'Группа', 'Тип Исследования', 'Часть тела',
    'Модальность снимка', 'Класс', 'Протокол',
    'Набор снимков', 'Укладка', 'Плотность железы', 'Ассиметрия',
    'Кальцинаты', 'Локальная перестройка структуры МЖ', 'Образования',
    'Инородные тела', 'Очаговые тени'
]


PROTOCOL = {
    "nabor-snimkov": {
        "label": "Набор снимков",
        "type": "radio",
        "tab": "1",
        "required": 1,
        "mode": "",
        "options": {
            "1": "Полный",
            "2": "Неполный"
        }
    },
    "ukladka": {
        "label": "Укладка",
        "type": "radio",
        "tab": "1",
        "required": 1,
        "mode": "",
        "options": {
            "1": "Правильная",
            "2": "Неправильная"
        }
    },
    "plotnost-zhelezy": {
        "label": "Плотность железы",
        "type": "radio",
        "tab": "1",
        "required": 1,
        "mode": "",
        "options": {
            "1": "A",
            "2": "B",
            "3": "C",
            "4": "D"
        }
    },
    "assimetriya": {
        "label": "Ассиметрия",
        "type": "radio",
        "tab": "1",
        "required": 1,
        "mode": "",
        "options": {
            "1": "Нет",
            "2": "Есть Локальная (очаговая)",
            "3": "Есть Тотальная",
            "4": "Есть Локальная с кальцинатами",
            "5": "Есть Динамическая"
        }
    },
    "kalcinaty": {
        "label": "Кальцинаты",
        "type": "radio",
        "tab": "1",
        "required": 1,
        "mode": "",
        "options": {
            "1": "Нет",
            "2": "Есть Доброкачественные",
            "3": "Есть Злокачественные"
        }
    },
    "lokalnaya-perestrojka-struktury-mzh": {
        "label": "Локальная перестройка структуры МЖ",
        "type": "radio",
        "tab": "1",
        "required": 1,
        "mode": "soft",
        "options": {
            "1": "Нет",
            "2": "Есть"
        }
    },
    "obrazovaniya": {
        "label": "Образования",
        "type": "checkboxlist",
        "tab": "1",
        "required": 1,
        "mode": "soft",
        "options": {
            "1": "Нет",
            "2": "Есть Однородной структуры",
            "3": "Есть Неоднородной структуры",
            "4": "Есть с четкими ровными контурами",
            "5": "Есть с нечеткими неровными контурами",
            "6": "Есть Правильной формы",
            "7": "Есть Неправильной формы",
            "8": "Есть без кальцинатов",
            "9": "Есть с кальцинатами"
        }
    },
    "inorodnye-tela-4": {
        "label": "Инородные тела",
        "type": "checkboxlist",
        "tab": "1",
        "required": 1,
        "mode": "soft",
        "options": {
            "1": "Нет",
            "2": "Есть Накожные метки",
            "3": "Есть импланты МЖ",
            "4": "Есть Внутритканевые метки"
        }
    },
    "ochagovye-teni": {
        "label": "Очаговые тени",
        "type": "labelpoly",
        "tab": "1",
        "required": 1,
        "mode": "",
        "options": []
    }
}
