import os
import torch
import torchvision
import numpy as np
from pytorch_lightning.callbacks.base import Callback
from import log_histogram_3d

class Comet3DHistogramSampler(Callback):
    def __init__(self, _C,logger):
        super().__init__()
        self.logger = logger

    def on_train_epoch_end(self, trainer, pl_module):
    # def on_after_backward(self, trainer, pl_module):

        model = pl_module.trainer.model
        epoch = model.current_epoch

        if trainer.global_step % 5 == 0:  # don't make the file huge
            params = model.state_dict()
            for k, v in params.items():
                grads = v
                name = f"epoch_{epoch}_params/{k}"
                self.logger.experiment.add_histogram(tag=name, values=grads,
                                                    global_step=self.trainer.global_step)

                name = f"epoch_{epoch}_params/{k}"
                self.logger.experiment.add_histogram(tag=name, values=grads.grad,
                                                    global_step=self.trainer.global_step)

        # Add this to the training loop in the train_one_epoch function
        # for name, param in model.named_parameters():
        #         log_histogram_3d(
        #             values=param,
        #             name=f"epoch_{epoch}_params/{name}",
        #             step=epoch
        #         )

        #         log_histogram_3d(
        #             values=param.grad,
        #             name=f"epoch_{epoch}_grads/{name}",
        #             step=epoch
        #         )

        # For run parameters - 
        # Add this to end of epoch loop
    # def on_train_epoch_end(self, trainer, pl_module):
    #     for layer, (name, param) in enumerate(model.named_parameters()):
    #             writer.add_histogram(
    #                 tag=f"run_params/{name}", values=param, global_step=epoch
    #             )