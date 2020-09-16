from pytorch_lightning.callbacks import ModelCheckpoint

# DEFAULTS used by the Trainer


def CheckpointCallback(_C):
    checkpoint_callback = ModelCheckpoint(
        filepath=str(_C.OUTPUT.CHECKPOINT_ROOT),
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
        prefix="",
    )

    return checkpoint_callback
