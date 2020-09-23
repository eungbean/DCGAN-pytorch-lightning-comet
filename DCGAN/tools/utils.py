import os
import random

import torch
import torch.nn as nn
import torchvision


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def experiment_subdir(exp_title, exp_id):
    assert 0 <= exp_id <= 9999
    return f"{exp_title}_{exp_id:04d}"


def save_img_grid(self, _C, imgs, path):
    self.grid_path = os.path.join(
        _C.OUTPUT.PREDICTION_DIR, f"epoch {self.current_epoch}.png"
    )
    grid = torchvision.utils.make_grid(imgs, nrow=10)
    torchvision.utils.save_image(grid, path)


def set_random_seed():
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    print("Random Seed: ", manualSeed)


def make_output_folders(_C):

    folders = [
        _C.OUTPUT.LOG_DIR,
    ]
    if _C.SAVE_CHECKPOINTS:
        folders.append(_C.OUTPUT.CHECKPOINT_DIR)
    if _C.SAVE_PREDICTIONS:
        folders.append(_C.OUTPUT.PREDICTION_DIR)

    if _C.SAVE_PREDICTIONS:
        folders.append(_C.OUTPUT.PREDICTION_DIR+'_norm')

    for folder in folders:
        os.makedirs(folder, exist_ok=True)
