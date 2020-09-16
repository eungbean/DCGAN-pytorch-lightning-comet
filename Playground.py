from yacs.config import CfgNode as CN
from pathlib import Path

_C = CN()

BASE_DIR = Path.cwd()
_C.BASE_DIR = str(BASE_DIR)
print(_C.BASE_DIR)
