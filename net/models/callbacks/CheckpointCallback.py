from pytorch_lightning.callbacks import ModelCheckpoint


# DEFAULTS used by the Trainer
def CheckpointCallback(_C):
    if _C.OUTPUT.SAVE_CHECKPOINTS:
        checkpoint_callback = ModelCheckpoint(
            filepath=str(_C.OUTPUT.CHECKPOINT_DIR),
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min",
            prefix="",
        )
    else:
        checkpoint_callback = None

    return checkpoint_callback
