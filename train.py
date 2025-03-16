import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from lit_resnet import LitResnet
# from wsi_dataset import create_dataloader as create_wsi_dataloader
# from infinite_tile_dataset import create_dataloader as create_infinite_tile_dataloader
# from tile_dataset import create_dataloader as create_tile_dataloader
# import argparse
from callbacks import EpochTimerCallback
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import call, instantiate
import torch
import os
from datetime import datetime
import logging 
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logger.debug(f"Trainer - main started:\n{cfg}")
    # Data loaders
    train_dataloader = call(cfg.data.train_dataloader)
    val_dataloader = call(cfg.data.val_dataloader)

    # Model
    logger.info("Constructing model")
    model = LitResnet(**cfg.model)

    # Callbacks
    checkpoint_dir = cfg.general.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoints_cb = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        filename='tupac-{epoch:02d}-{val_loss:.2f}',
        save_top_k=2,
        mode='min',
        save_last=True,
    )
    early_stopping_cb = EarlyStopping(monitor='val_loss',patience=5, mode='min')
    epoch_time_cb = EpochTimerCallback()
    # Logger
    tb_logger = TensorBoardLogger(checkpoint_dir, name="dig_patho")# , prefix=timestamp)

    # Trainer
    logger.info(f"Initializing Trainer")
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        callbacks=[checkpoints_cb, early_stopping_cb, epoch_time_cb],
        logger=tb_logger,
        # To use with the infinite dataloader
        # limit_train_batches=100, limit_val_batches=10,
        # gpus=1 if torch.cuda.is_available() else 0,  accelerator='dp' if torch.cuda.is_available() else None
    )

    # Train
    trainer.fit(model, train_dataloader, val_dataloader)
    logger.info(f"Train completed, Saving final config to file")
    OmegaConf.save(cfg, os.path.join(checkpoint_dir, "config.yaml"))

if __name__ == '__main__':
    logger = logging.getLogger("Trainer")
    logger.setLevel(logging.DEBUG)
    main()
