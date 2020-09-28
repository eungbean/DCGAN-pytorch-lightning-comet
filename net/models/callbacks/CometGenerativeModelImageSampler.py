import os
import torch
import torchvision
import numpy as np
from pytorch_lightning.callbacks.base import Callback


class CometGenerativeModelImageSampler(Callback):
    def __init__(self, _C, n_samples=100, logger=None):
        super().__init__()
        self.n_samples = n_samples
        self.NUM_Z = _C.MODEL.NUM_Z
        self.SAVE_PRED_NORM = _C.OUTPUT.SAVE_PRED_NORM

        self.PREDICTION_DIR = _C.OUTPUT.PREDICTION_DIR
        self.comet_logger = logger

    def on_train_start(self, trainer, pl_module):

        self.PRED_DIR = os.path.join(
            self.PREDICTION_DIR,
            "epoch 0_initial.png",
        )
        self.PRED_DIR_NORM = os.path.join(
            self.PREDICTION_DIR + "_norm",
            "epoch 0_initial.png",
        )

        # TODO Dimension variations
        self.z_log = torch.randn(
            self.n_samples, self.NUM_Z, 1, 1, device=pl_module.device
        )

        with torch.no_grad():
            pl_module.eval()
            images = pl_module(self.z_log)
            pl_module.train()

        # save original image
        save_img_grid(imgs=images, PREDICTION_DIR=self.PRED_DIR)
        self.comet_logger.experiment.log_image(
            image_data=self.PRED_DIR_NORM,
            name="Training_normalized",
            image_channels="first",
            step=0,
        )

        # save normalized image
        if self.SAVE_PRED_NORM:
            save_img_grid(
                imgs=images, PREDICTION_DIR=self.PRED_DIR_NORM, normalize=True
            )
            self.comet_logger.experiment.log_image(
                image_data=self.PRED_DIR,
                name="Training",
                image_channels="first",
                step=0,
            )

    def on_epoch_end(self, trainer, pl_module):

        self.PRED_DIR = os.path.join(
            self.PREDICTION_DIR,
            f"epoch {pl_module.trainer.model.current_epoch+1}.png",
        )
        self.PRED_DIR_NORM = os.path.join(
            self.PREDICTION_DIR + "_norm",
            f"epoch {pl_module.trainer.model.current_epoch+1}.png",
        )

        # generate images
        with torch.no_grad():
            pl_module.eval()
            images = pl_module(self.z_log)
            pl_module.train()

        # save original image
        save_img_grid(imgs=images, PREDICTION_DIR=self.PRED_DIR, normalize=False)
        self.comet_logger.experiment.log_image(
            image_data=self.PRED_DIR,
            name="Training",
            image_channels="first",
            step=pl_module.trainer.model.current_epoch + 1,
        )


        # save normalized image
        if self.SAVE_PRED_NORM:
            save_img_grid(imgs=images, PREDICTION_DIR=self.PRED_DIR_NORM, normalize=True)
            self.comet_logger.experiment.log_image(
                image_data=self.PRED_DIR_NORM,
                name="Training_normalized",
                image_channels="first",
                step=pl_module.trainer.model.current_epoch + 1,
            )



def save_img_grid(imgs, PREDICTION_DIR, normalize=False):
    nrow = np.int(np.round(np.sqrt(len(imgs))))
    grid = torchvision.utils.make_grid(imgs, nrow)
    torchvision.utils.save_image(grid, fp=PREDICTION_DIR, normalize=normalize)
