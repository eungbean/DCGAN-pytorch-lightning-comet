import os
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


def save_img_grid(self,_C):
    self.grid_path = os.path.join(
        _C.OUTPUT.PREDICTION_ROOT, f"epoch {self.current_epoch}.png"
        )
    grid = torchvision.utils.make_grid(self.g_img_log, nrow=10)
    torchvision.utils.save_image(grid, self.grid_path)