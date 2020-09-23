# fmt: off
# flake8: noqa
from pl_bolts import datamodules
from torchvision import transforms

def CIFAR10(_C):
    cifar10_transforms = transforms.Compose(
        [
            transforms.Resize(_C.INPUT.IMGSIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    
# fmt: off
    return datamodules.CIFAR10DataModule(
        data_dir        =_C.INPUT.DATASET_DIR,
        val_split       =5000,
        num_workers     =_C.DATALOADER.NUM_WORKERS,
        batch_size      =_C.DATALOADER.BATCH_SIZE,
        train_transforms=cifar10_transforms,
        test_transforms =cifar10_transforms,
        val_transforms  =cifar10_transforms,
    )
