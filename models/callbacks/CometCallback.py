from pathlib import Path

import torchvision
from pytorch_lightning.callbacks.base import Callback

p = Path(".")


class CometCallback(Callback):
    def __init__(self):
        print("Comet Callback initialized")

    def on_epoch_end(self):
        # Mage grid of generated samples
        grid_path = p / "output" / "grids" / f"epoch {self.current_epoch}.png"
        grid = torchvision.utils.make_grid(self.fake[:8], nrow=4)

        # Log generated images
        torchvision.utils.save_image(grid, grid_path)
        self.experiment.experiment.log_image(
            image_data=grid_path,
            name=f"Training",
            image_channels="first",
        )
