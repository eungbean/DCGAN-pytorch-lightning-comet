import torch
import torch.nn as nn

import torchvision
from torch.optim import Adam
import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchgan
from torchgan.models import DCGANGenerator, DCGANDiscriminator
from torchgan.losses import MinimaxGeneratorLoss, MinimaxDiscriminatorLoss
from torchgan.trainer import Trainer

from tools.logger import comet_logger
from models.configs import load_default_config
from data import get_cifar10

# Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
print("Random Seed: ", manualSeed)

# load configurations
_C = load_default_config()
_C.merge_from_file(".private/secrets.yaml")
_C.freeze()

# # Define Logger
# comet_logger = get_comet_logger(_C)
# comet_logger.experiment.log_parameters(_C)

# CUDA
device = torch.device("cuda")
# Use deterministic cudnn algorithms
torch.backends.cudnn.deterministic = True


# dataset
def cifar10():
    train_dataset = dsets.CIFAR10(
        root="./dataset/CIFAR10",
        train=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        ),
        download=True,
    )
    train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    return train_loader


dcgan = {
    "generator": {
        "name": DCGANGenerator,
        "args": {
            "encoding_dims": 100,
            "out_channels": 3,
            "step_channels": 16,
        },
        "optimizer": {"name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}},
    },
    "discriminator": {
        "name": DCGANDiscriminator,
        "args": {"in_channels": 3, "step_channels": 16},
        "optimizer": {
            "name": Adam,
            "args": {
                "lr": 0.0002,
                "betas": (0.5, 0.999),
            },
        },
    },
}

loss = [MinimaxGeneratorLoss(), MinimaxDiscriminatorLoss()]

trainer = Trainer(
    dcgan,
    loss,
    sample_size=64,
    epochs=100,
    checkpoints=_C.OUTPUT.CHECKPOINT_DIR,
    recon=_C.OUTPUT.PREDICTION_DIR,
    log_dir=_C.OUTPUT.LOG_DIR,
    device=device,
)

trainer(cifar10())
