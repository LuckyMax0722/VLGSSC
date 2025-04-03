import os
import sys
from easydict import EasyDict

CONF = EasyDict()

# Path
CONF.PATH = EasyDict()

## data
CONF.PATH.DATA_ROOT = '/u/home/caoh/datasets/SemanticKITTI/dataset' # TODO: Change path to your VLGSSC-dir
CONF.PATH.DATA_TEXT = os.path.join(CONF.PATH.DATA_ROOT, 'text')
CONF.PATH.DATA_TEXT_FEAT = os.path.join(CONF.PATH.DATA_TEXT, 'feat')

