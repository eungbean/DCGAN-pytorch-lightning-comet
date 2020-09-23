# fmt: off
# flake8: noqa
from pl_bolts.datamodules import CIFAR10DataModule

def MNIST(_C):
    return CIFAR10DataModule(
        self,
        data_dir    : str = None,
        val_split   : int = 5000,
        num_workers : int = 16,
        batch_size  : int = 32,
        seed        : int = 42,
        )