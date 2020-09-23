import os
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl

from data import CIFAR10

from models import Generator
from models import Discriminator
from models.metrics.accuracy_CIFAR10 import accuracy
from models.metrics.zeroCenteredGP import zeroCenteredGP

from models.configs import load_default_config
from tools.logger import get_comet_logger
from models.callbacks import (
    CometGenerativeModelImageSampler
    )
from tools.utils import (
    weights_init,
    set_random_seed,
    make_output_folders,
)


# Set random seed for reproducibility
set_random_seed()

# load configurations
_C = load_default_config()
_C.merge_from_file(os.path.join("models", "configs", "DCGAN-ZeroCenteredGP.yaml"))
_C.freeze()

# Define Logger
comet_logger = get_comet_logger(_C)

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
        self.gen_img = None

        ## Initialize Weight
        self.netD.apply(weights_init)
        self.netG.apply(weights_init)

    # 2. Computations (forward)
    def forward(self, z):
        return self.netG(z)

    # Define Loss
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    # 3. What happens inside the training loop (training_step)
    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch
        b_size = imgs.shape[0]
        
        ## Update Generator network
        if optimizer_idx == 0:
            # Sample noise
            z = torch.randn(b_size, _C.MODEL.NUM_Z, 1, 1, device=self.device)
            z = z.type_as(imgs)

            ## Generate fake images
            self.gen_img = self.netG(z)

            ## ground truth result (ie: all real)
            label_real = torch.full((b_size,), self.real_label, device=self.device)
            label_real = label_real.type_as(imgs)

            ## adversarial loss is binary cross-entropy
            self.predicts = self.netD(self.gen_img.type_as(imgs))
            errG = self.adversarial_loss(self.predicts.view(-1), label_real)

            # Log and output the progress
            tqdm_dict = {"g_loss": errG}
            output = OrderedDict(
                {"loss": errG, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )

            return output

        ## train discriminator: Measure discriminator's ability to classify real from generated samples
        if optimizer_idx == 1:
            
            self.netD.zero_grad()

            # 텐서에 행해지는 모든 연산에 대한 미분값을 계산
            imgs.requires_grad_()
            
            # D(X) : Train with all-real batch: how well can it label as real?
            label_real = torch.full((b_size,), self.real_label, device=self.device)
            label_real = label_real.type_as(imgs)

            # D(G(z))
            pred_real = self.netD(imgs).view(-1)
            errD_real = self.adversarial_loss(pred_real, label_real)

            # D(G(Z)) : Train with all-fake batch: how well can it label as fake?
            label_fake = torch.full((b_size,), self.fake_label, device=self.device)
            label_fake = label_fake.type_as(imgs)

            pred_fake = self.netD(self.gen_img.detach()).view(-1)
            errD_fake = self.adversarial_loss(pred_fake, label_fake)

            # # Gradient Panelty
            # gradient_panelty = self.get_gradient_penelty(imgs ,pred_fake)

            # Zero-centered Gradient Panelty
            zero_centered_gp= zeroCenteredGP(imgs, pred_real)


            # discriminator loss is the average of these
            errD = (errD_real + errD_fake * zero_centered_gp) / 2

            # Accuracy metric
            d_acc_real = accuracy(pred_real, label_fake)
            d_acc_fake = accuracy(pred_fake, label_fake)
            d_acc = (d_acc_real + d_acc_fake) / 2

            # Log and output the progress
            tqdm_dict = {
                "d_loss": errD,
                "D_acc": d_acc,
            }

            log_dict = {
                "d_loss": errD,
                "D_acc_real": d_acc_real,
                "D_acc_fake": d_acc_fake,
                "D_acc": d_acc,
            }

            output = OrderedDict(
                {"loss": errD, "progress_bar": tqdm_dict, "log": log_dict}
            )

            return output

    def configure_optimizers(self):
        optimizerD = Adam(
            self.netD.parameters(),
            lr=_C.SOLVER.D_LR,
            betas=(_C.SOLVER.BETA1, _C.SOLVER.BETA2),
        )
        optimizerG = Adam(
            self.netG.parameters(),
            lr=_C.SOLVER.G_LR,
            betas=(_C.SOLVER.BETA1, _C.SOLVER.BETA2),
        )
        return [optimizerG, optimizerD], []


# Define model
model = DCGAN()

# Make Output Folders
make_output_folders(_C)

# Pytorch-Lightening Trainer
trainer = pl.Trainer(
    min_epochs=1,
    max_epochs=_C.SOLVER.EPOCHS,
    logger=comet_logger,
    callbacks=[
        CometGenerativeModelImageSampler(_C, 100, comet_logger),
    ],
    gpus=_C.SYSTEM.NUM_GPU_WORKER,
)

# Let's Train!
trainer.fit(
    model,
    CIFAR10(_C),
)

# trainer.save_checkpoint(os.path.join(_C.OUTPUT.CHECKPOINT_DIR, "DCGAN.ckpt")
