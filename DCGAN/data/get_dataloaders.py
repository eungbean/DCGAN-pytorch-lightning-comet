import numpy as np
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision.datasets.folder import ImageFolder

from .transformation import get_augmentation, get_preprocess


def get_dataloaders(_C):
    """"""

    # TODO Apply Preprocess
    # preprocessing_train = get_preprocess(_C, is_test=False)
    # preprocessing_test  = get_preprocess(_C, is_test=True)
    # augmentation        = get_augmentation(_C, is_train=True)
    augmentation = None

    # create the datasets
    # train_ds    = ImageFolder(root=_C.INPUT.TRAIN_DIR, transform=augmentation)
    # val_ds      = ImageFolder(root=_C.INPUT.VAL_DIR, transform=augmentation)

    # Bue we'll use public torchvision.datasets for simplicity.
    train_ds = torchvision.datasets.CIFAR10(
        root=_C.INPUT.DATASET_DIR,
        train=True,
        transform=augmentation,
        target_transform=None,
        download=True,
    )

    # test_ds = torchvision.datasets.CIFAR10(
    #     root=_C.INPUT.DATASET_DIR,
    #     train=False,
    #     transform=augmentation,
    #     target_transform=None,
    #     download=True,
    # )

    # logging.info(f'Train samples={len(train_ds)}, Validation samples={len(val_ds)}, Test samples={len(test_ds)}')

    train_dl = DataLoader(
        train_ds,
        batch_size=_C.DATALOADER.BATCH_SIZE,
        num_workers=_C.DATALOADER.NUM_WORKERS,
        shuffle=_C.DATALOADER.TRAIN_SHUFFLE,
    )
    # *args, **kwargs)

    # val_dl = DataLoader(
    #     val_ds,
    #     batch_size  =_C.DATALOADER.BATCH_SIZE,
    #     num_workers =_C.DATALOADER.NUM_WORKERS,
    #     shuffle     =False,
    #     *args, **kwargs)

    # test_dl = DataLoader(
    #     test_ds,
    #     batch_size=_C.DATALOADER.BATCH_SIZE,
    #     num_workers=_C.DATALOADER.NUM_WORKERS,
    #     shuffle=_C.DATALOADER.TEST_SHUFFLE,
    # )

    return train_dl, test_dl
