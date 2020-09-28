# fmt: off
# flake8: noqa
from yacs.config import CfgNode as CN
from pathlib import Path
import datetime
from pytz import timezone

def timestamp(folder_name):
    return datetime.datetime.now().astimezone(timezone(TIMEZONE)).strftime(folder_name)

"""
Config with YACS and Path library to config our model.
YCAS: https://github.com/rbgirshick/yacs
"""

# Set your timeZone
TIMEZONE = 'Asia/Seoul'             #Default=Your computer Setting

_C = CN()
_C.EXP_TITLE = "DCGAN"
_C.MAINTAINER = "Eungbean Lee"

# SYSTEM
_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPU_WORKER = -1       # Default(-1)=all gpu
_C.SYSTEM.DATAPARALLEL   = 'dp'     # ['dp', 'ddp']

# Input
_C.INPUT = CN()
_C.INPUT.DATASET = "CIFAR10"        #['CIFAR10', 'CELEBA']
_C.INPUT.IMGSIZE = 64
_C.INPUT.TRAIN_VAL_SPLIT = 0.1

# Directories
BASE_DIR                    = Path.cwd()
DATASET_DIR                 = BASE_DIR / "dataset" / _C.INPUT.DATASET

_C.BASE_DIR                 = str(BASE_DIR)
_C.INPUT.DATASET_DIR        = str(DATASET_DIR)
_C.INPUT.TRAIN_DIR          = str(DATASET_DIR / "train")
_C.INPUT.VAL_DIR            = str(DATASET_DIR / "val")
_C.INPUT.MEAN_STD_DIR       = str(DATASET_DIR / "img_mean_std")


# OUTPUT
OUTPUT_FOLDER              = _C.EXP_TITLE + '_' + _C.INPUT.DATASET + '_' + timestamp('%y%m%d_%H-%M-%S')
# OUTPUT_FOLDER              = _C.EXP_TITLE + '_' + _C.INPUT.DATASET + '_' + 'tmp'
_C.OUTPUT = CN()
_C.OUTPUT.LOG_DIR          = str(BASE_DIR / "output" / OUTPUT_FOLDER / "logs")
_C.OUTPUT.CHECKPOINT_DIR   = str(BASE_DIR / "output" / OUTPUT_FOLDER / "checkpoints")
_C.OUTPUT.PREDICTION_DIR   = str(BASE_DIR / "output" / OUTPUT_FOLDER / "predictions")
_C.OUTPUT.SAVE_CHECKPOINTS = False  # True
_C.OUTPUT.SAVE_PREDICTIONS = True
_C.OUTPUT.SAVE_PRED_NORM   = False
_C.EXP_ID                  = 9999  # 0~9999

# Model
_C.MODEL = CN()
_C.MODEL.NUM_Z      = 100
_C.MODEL.NUM_G_FEAT = 64
_C.MODEL.NUM_D_FEAT = 64
_C.MODEL.NUM_C      = 3


# Solver
_C.SOLVER = CN()
_C.SOLVER.EPOCHS    = 100
_C.SOLVER.G_LR      = 2e-4
_C.SOLVER.D_LR      = 2e-4
_C.SOLVER.OPTIMIZER = "adam"  # ['adam', 'adamw']
_C.SOLVER.BETA1     = 0.5
_C.SOLVER.BETA2     = 0.999


# Eval
_C.EVAL = CN()

# TRANSFORM
_C.TRANSFORM = CN()

# PROB = Probability of applying transform. 0 = Disable Transformation
## TRANSFORM - FLIP
_C.TRANSFORM.TRAIN_FLIP_ENABLE          = True
_C.TRANSFORM.TRAIN_HORIZONTAL_FLIP_PROB = 0.0
_C.TRANSFORM.TRAIN_VERTICAL_FLIP_PROB   = 0.0

## TRANSFORM - CROP
_C.TRANSFORM.TRAIN_RANDOM_CROP_PROB     = 0.0
_C.TRANSFORM.TRAIN_RANDOM_CROP_SIZE     = (_C.INPUT.IMGSIZE, _C.INPUT.IMGSIZE)

## TRANSFORM - ROTATE
_C.TRANSFORM.TRAIN_RANDOM_ROTATE_PROB   = 1.0
_C.TRANSFORM.TRAIN_RANDOM_ROTATE_DEG    = 45

## TRANSFORM - NOISE
_C.TRANSFORM.TRAIN_SPECKLE_NOISE_PROB   = 0.0
_C.TRANSFORM.TRAIN_SPECKLE_NOISE_STD    = 0.0

## TRANSFORM - BLUR
_C.TRANSFORM.TRAIN_BLUR_MOTION_PROB     = 2
_C.TRANSFORM.TRAIN_BLUR_ONEOF           = 0.2
_C.TRANSFORM.TRAIN_BLUR_MEDIAN_PROB     = 0
_C.TRANSFORM.TRAIN_BLUR_MEDIAN_LIMIT    = 3
_C.TRANSFORM.TRAIN_BLUR_PROB            = 0.1
_C.TRANSFORM.TRAIN_BLUR_LIMIT           = 3

## TRANSFORM - RANDOM BRIGHTNESS
_C.TRANSFORM.TRAIN_RANDOM_BRIGHT_PROB = 1.0
_C.TRANSFORM.TRAIN_RANDOM_BRIGHT_STD = 0.0
_C.TRANSFORM.TEST_SIZE                  = (32, 32)


# # Data loader
_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE     = 128
_C.DATALOADER.NUM_WORKERS    = 8  # Recommended: Half of total CPU cores.
_C.DATALOADER.TRAIN_SHUFFLE  = True
_C.DATALOADER.TEST_SHUFFLE   = True


# Comet Logger
_C.COMET = CN()
# _C.COMET.DISABLED       = True # Disable Commet for Test
_C.COMET.DISABLED       = False # Disable Commet for Test
_C.COMET.APIKEY         = "MY_COMET_APIKEY"
_C.COMET.PROJECT_NAME   = "MY_COMET_PROJECT_NAME"   # OPTIONAL
_C.COMET.WORKSPACE      = "MY_COMET_WORKSPACE"      # OPTIONAL
_C.COMET.REST_API_KEY   = "COMET_REST_KEY"          # OPTIONAL

## Configure a experiment on the Comet.ml frontend.
## https://www.comet.ml/docs/python-sdk/Experiment/
_C.COMET.LOG_STEP           = 100
_C.COMET.LOG_CODE           = True  # Default(True)
_C.COMET.LOG_GRAPH          = True  # Default(True)
_C.COMET.AUTO_LOG_WEIGHT    = True  # Default(False)
_C.COMET.AUTO_LOG_PARAM     = True  # Default(True)
_C.COMET.AUTO_LOG_METRIC    = True  # Default(True)
_C.COMET.AUTO_LOG_GRAPH     = True  # Default(False)
_C.COMET.AUTO_LOG_OUTPUT    = "default" # Default("default") ["native", "simple", "default", False]

_C.merge_from_file(str(BASE_DIR / "tools" /"logger"/"private.yaml"))

def get_default_config():
    return _C.clone()
