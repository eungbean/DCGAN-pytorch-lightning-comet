from yacs.config import CfgNode as CN
from pathlib import Path

"""
I use YACS and Path library to config our model.
YCAS: https://github.com/rbgirshick/yacs
"""

_C = CN()
_C.EXP_TITLE    = 'DCGAN'
_C.MAINTAINER   = 'Eungbean Lee'

# Input
_C.INPUT = CN()
_C.INPUT.DATASET                = 'CIFAR10'
_C.INPUT.IMGSIZE                  = 64
_C.INPUT.TRAIN_VAL_SPLIT          = 0.1

# Directories
BASE_DIR                  = Path.cwd()
DATASET_DIR               = BASE_DIR / 'dataset' / _C.INPUT.DATASET
_C.INPUT.TRAIN_DIR        = str(DATASET_DIR / 'train'       )
_C.INPUT.VAL_DIR          = str(DATASET_DIR / 'val'         )
_C.INPUT.MEAN_STD_DIR     = str(DATASET_DIR / 'img_mean_std')

# NOT CUSTOMIZABLE
_C.BASE_DIR               = str(BASE_DIR    )
_C.INPUT.DATASET_DIR      = str(DATASET_DIR )

# OUTPUT
_C.OUTPUT = CN()
_C.OUTPUT.LOG_ROOT        = str(BASE_DIR / 'output' / 'logs')
_C.OUTPUT.WEIGHT_ROOT     = str(BASE_DIR / 'output' / 'weights')
_C.OUTPUT.CHECKPOINT_ROOT = str(BASE_DIR / 'output' / 'checkpoints')
_C.OUTPUT.PREDICTION_ROOT = str(BASE_DIR / 'output' / 'predictions')
_C.SAVE_CHECKPOINTS       = False # True
_C.EXP_ID                 = 9999  # 0~9999

# Model
_C.MODEL = CN()
_C.MODEL.NUM_Z      = 100
_C.MODEL.NUM_G_FEAT = 64
_C.MODEL.NUM_D_FEAT = 64
_C.MODEL.NUM_C      = 3


# # Solver
_C.SOLVER = CN()
_C.SOLVER.EPOCHS                 = 500
_C.SOLVER.G_LR                   = 2e-4
_C.SOLVER.D_LR                   = 1e-3
_C.SOLVER.OPTIMIZER              = 'adam'  # ['adam', 'adamw']
_C.SOLVER.BETA1                  = 0.5


# Eval
_C.EVAL             = CN()


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
_C.TRANSFORM.TRAIN_RANDOM_BRIGHTNESS_PROB   = 1.0
_C.TRANSFORM.TRAIN_RANDOM_BRIGHTNESS_STD    = 0.0
_C.TRANSFORM.TEST_SIZE  = (32,32)


# # Data loader
_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE        = 128
_C.DATALOADER.NUM_WORKERS       = 8     # Recommended: Half of total CPU cores.
_C.DATALOADER.TRAIN_SHUFFLE     = True
_C.DATALOADER.TEST_SHUFFLE      = True


#Comet
_C.COMET = CN()
_C.COMET.APIKEY         = "MY_COMET_APIKEY"
_C.COMET.PROJECT_NAME   = "MY_COMET_PROJECT_NAME" #OPTIONAL
_C.COMET.WORKSPACE      = "MY_COMET_WORKSPACE" #OPTIONAL
# _C.COMET.REST_API_KEY   = "COMET_REST_KEY" #OPTIONAL


# MISC
_C.MISC = CN()
_C.MISC.NUM_GPU_WORKER      = 1

def get_default_config():
    return _C.clone()