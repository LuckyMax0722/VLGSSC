import os
import sys
from easydict import EasyDict

CONF = EasyDict()

# Path
CONF.PATH = EasyDict()
CONF.PATH.BASE = '/u/home/caoh/projects/MA_Jiachen/3DPNA'  # TODO: Change path to your SGN-dir

## data
CONF.PATH.DATA_ROOT = '/u/home/caoh/datasets/SemanticKITTI/dataset'
CONF.PATH.DATA_TEXT = os.path.join(CONF.PATH.DATA_ROOT, 'text')

## log
CONF.PATH.LOG_DIR = os.path.join(CONF.PATH.BASE, 'output')

## ckpt
CONF.PATH.CKPT_DIR = os.path.join(CONF.PATH.BASE, 'ckpts')

## config
CONF.PATH.CONFIG_DIR = os.path.join(CONF.PATH.BASE, 'configs')

