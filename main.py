import os
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
import torchvision


from data import get_cifar10

from models import Generator
from models import Discriminator

from models.callbacks import CheckpointCallback
from models.configs import load_config
from tools.logger import get_cometLogger
from tools.utils import weights_init, save_img_grid


# load configurations
_C = load_config()
_C.merge_from_file(".private/secrets.yaml")
_C.freeze()

"""
Each project goes into a LightningModule.
This module houses:
1. Model definition (__init__)
2. Computations (forward)
3. What happens inside the training loop (training_step)
4. What happens inside the validation loop (validation_step)
5. What optimizer(s) to use (configure_optimizers)
6. What data to use (train_dataloader, val_dataloader, test_dataloader)
"""
# Define Logger
cometLogger = get_cometLogger(_C)
cometLogger.experiment.log_parameters(_C)
class DCGAN(pl.LightningModule):

    # 1. Model definition (__init__)
    def __init__(self):
        super(DCGAN, self).__init__()

        # Define Network
        self.netG = Generator(_C)
        self.netD = Discriminator(_C)

        ## Define labels
        self.real_label = 1
        self.fake_label = 0
        self.gen_img =  None

        # Define Z for log to maintain consistency through whole experiments
        self.z_log = torch.randn(100, _C.MODEL.NUM_Z, 1, 1, device=self.device)

        if _C.MISC.NUM_GPU_WORKER == 0:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
            self.netD = nn.DataParallel(self.netD.to(device))
            self.netG = nn.DataParallel(self.netG.to(device))

        ## Initialize Weight
        self.netD.apply(weights_init)
        self.netG.apply(weights_init)

    # 2. Computations (forward)
    def forward(self, z):
        return self.netG(z)
    
    # Define Loss
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    # 3. What happens inside the training loop (training_step)
    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch
        b_size = imgs.shape[0]

        if self.current_epoch==0:
            self.z_log = self.z_log.type_as(imgs)


        ## Update Generator network
        if optimizer_idx == 0:
            #Sample noise
            z = torch.randn(b_size, _C.MODEL.NUM_Z, 1, 1, device=self.device)
            z = z.type_as(imgs)

            ## Generate fake images
            self.gen_img =  self.netG(z)

            ## ground truth result (ie: all real)
            label_real = torch.full((b_size,), self.real_label, device=self.device)
            label_real = label_real.type_as(imgs)

            ## adversarial loss is binary cross-entropy
            self.predicts = self.netD(self.gen_img.type_as(imgs))
            errG = self.adversarial_loss(self.predicts.view(-1), label_real)

            tqdm_dict = {"g_loss": errG}
            output = OrderedDict(
                {"loss": errG, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )

            return output

        ## train discriminator: Measure discriminator's ability to classify real from generated samples
        if optimizer_idx == 1:

            ## Train with all-real batch: how well can it label as real?
            label_real = torch.full((b_size,), self.real_label, device=self.device)
            label_real = label_real.type_as(imgs)
            
            pred_real = self.netD(imgs)
            errD_real = self.adversarial_loss(pred_real.view(-1), label_real)

            ## Train with all-fake batch: how well can it label as fake?
            label_fake = torch.full((b_size,), self.fake_label, device=self.device)
            label_fake = label_fake.type_as(imgs)
            
            pred_fake = self.netD(self.gen_img.detach())
            errD_fake = self.adversarial_loss(pred_fake.view(-1), label_fake)
            errD = (errD_real + errD_fake) / 2  ## discriminator loss is the average of these
            
            pred_real = torch.round(pred_real)
            pred_fake = torch.round(pred_fake)
            accD_real = accuracy(pred_real.view(-1), label_real, num_classes=2)
            accD_fake = accuracy(pred_fake.view(-1), label_fake, num_classes=2)
            
            tqdm_dict = {"d_loss": errD,"D_acc_fake":accD_fake, "D_acc_real":accD_real}        ## Log and output the progress
            
            output = OrderedDict(
                {"loss": errD, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )

            return output

    def on_epoch_end(self):
        # Save image grid every 10 epoches
        # if self.current_epoch% 10 == 0:

        self.g_img_log = self.netG(self.z_log)

        save_img_grid(self, _C)
        cometLogger.experiment.log_image(
            image_data=self.grid_path, 
            name=f"Training", image_channels="first"
            )

    # # What happens inside the validation loop (validation_step)
    # # But, How do we define validation loop in GAN?
    # def validation_step(self, batch, batch_idx, optimizer_idx):       
    #     pass # OPTIONAL

    # def validation_epoch_end(self, outputs):
    #     pass # OPTIONAL

    # def test_step(self, batch, batch_nb):
    #     pass # OPTIONAL

    # def test_epoch_end(self, output):
    #     pass #OPTIONAL

    # What optimizer(s) to use (configure_optimizers)
    def configure_optimizers(self):
        optimizerD = Adam(
            self.netD.parameters(), lr=_C.SOLVER.D_LR, betas=(_C.SOLVER.BETA1, 0.999)
        )
        optimizerG = Adam(
            self.netG.parameters(), lr=_C.SOLVER.G_LR, betas=(_C.SOLVER.BETA1, 0.999)
        )
        return [optimizerG, optimizerD], []

    # # What data to use (dataloader)
    # def train_dataloader(self):
    #     pass

    # def val_dataloader(self):
    #     pass

    # def test_dataloader(self):
    #     pass


# Define model
model = DCGAN()

cometLogger.experiment.set_model_graph(str(model))

# Pytorch-Lightening Trainer
trainer = pl.Trainer(
    gpus=_C.MISC.NUM_GPU_WORKER,
    min_epochs=1,
    max_epochs=_C.SOLVER.EPOCHS,
    logger=cometLogger,
    checkpoint_callback=CheckpointCallback(_C),
)

# Let's Train!
trainer.fit(
    model,
    get_cifar10(_C),
)
trainer.save_checkpoint(os.path.join(_C.OUTPUT.CHECKPOINT_ROOT,"DCGAN.ckpt"))