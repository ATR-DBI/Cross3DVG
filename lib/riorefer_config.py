import os
import sys
from easydict import EasyDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

CONF = EasyDict()

# path
CONF.PATH = EasyDict()
CONF.PATH.BASE = ROOT_DIR
CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, "data")

CONF.PATH.RIO = os.path.join(CONF.PATH.DATA, "rio")
CONF.PATH.LIB = os.path.join(CONF.PATH.BASE, "lib")
CONF.PATH.MODELS = os.path.join(CONF.PATH.BASE, "models")
CONF.PATH.UTILS = os.path.join(CONF.PATH.BASE, "utils")

# RIO data
CONF.PATH.SCANS = os.path.join(CONF.PATH.RIO, "scans")
CONF.PATH.META = os.path.join(CONF.PATH.RIO, "meta_data")
#CONF.PATH.SCAN_DATA = os.path.join(CONF.PATH.RIO, "rio_data")
CONF.PATH.SCAN_DATA = os.path.join(CONF.PATH.RIO, "rio200_data")

# data
CONF.RIO_DIR =  os.path.join(CONF.PATH.RIO, "scans")
CONF.SCENE_NAMES = sorted(os.listdir(CONF.RIO_DIR))
CONF.NYU40_LABELS = os.path.join(CONF.PATH.META, "nyu40_labels.csv")

# RIO
CONF.RIO_LIST = os.path.join(CONF.PATH.META, "scans.txt")

# clip
# text_feature for clip_loss
CONF.PATH.TEXT_FEATURE = os.path.join(CONF.PATH.DATA, 'riorefer/text_feature')

# image_feature for clip_loss
CONF.PATH.IMAGE_FEATURE = os.path.join(CONF.PATH.DATA, 'riorefer/image_feature')

# frame_square (frames_square/scene0000_00/pose)
#CONF.PATH.FRAME_SQUARE = os.path.join(CONF.PATH.DATA, "rio/frames_square")

# output
CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, "outputs")

# train
CONF.TRAIN = EasyDict()
CONF.TRAIN.MAX_DES_LEN = 126  # max description length
CONF.TRAIN.SEED = 42

