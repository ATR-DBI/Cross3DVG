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

CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA, "scannet")
CONF.PATH.LIB = os.path.join(CONF.PATH.BASE, "lib")
CONF.PATH.MODELS = os.path.join(CONF.PATH.BASE, "models")
CONF.PATH.UTILS = os.path.join(CONF.PATH.BASE, "utils")

# scannet data
CONF.PATH.SCANS = os.path.join(CONF.PATH.SCANNET, "scans")
CONF.PATH.META = os.path.join(CONF.PATH.SCANNET, "meta_data")
CONF.PATH.SCAN_DATA = os.path.join(CONF.PATH.SCANNET, "scannet200_data")

# data
CONF.SCANNET_DIR =  os.path.join(CONF.PATH.SCANNET, "scans") 
CONF.SCENE_NAMES = sorted(os.listdir(CONF.SCANNET_DIR))
CONF.NYU40_LABELS = os.path.join(CONF.PATH.META, "nyu40_labels.csv")

# scannet
CONF.SCANNETV2_LIST = os.path.join(CONF.PATH.META, "scannetv2.txt")

# clip
CONF.PATH.TEXT_FEATURE = os.path.join(CONF.PATH.DATA, 'scanrefer/text_feature')

# image_feature for clip_loss
CONF.PATH.IMAGE_FEATURE = os.path.join(CONF.PATH.DATA, 'scanrefer/image_feature')

# frame_square (frames_square/scene0000_00/pose)
#CONF.PATH.FRAME_SQUARE = os.path.join(CONF.PATH.DATA, "scannet/frames_square")

# output
CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, "outputs")

# train
CONF.TRAIN = EasyDict()
CONF.TRAIN.MAX_DES_LEN = 126  # max description length
CONF.TRAIN.SEED = 42
