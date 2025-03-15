import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from lit_resnet import LitResnet
from wsi_dataset import create_dataloader as create_wsi_dataloader
from infinite_tile_dataset import create_dataloader as create_infinite_tile_dataloader
from tile_dataset import create_dataloader as create_tile_dataloader
import argparse
import torch
import os
from datetime import datetime
import logging 
logger = logging.getLogger(__name__)

def main(args):
    # Data loaders
    train_dir = os.path.join('data/datasets', args.train_path)
    val_dir = os.path.join('data/datasets', args.val_path)
    logger.info(f"Construction DataLoader with train: {train_dir}, val: {val_dir}")
    # train_dataloader = create_wsi_dataloader(
    #     csv_file=train_csv, data_dir='data/raw/training_png', batch_size=args.batch_size, is_train=True
    # )
    # val_dataloader = create_wsi_dataloader(
    #     csv_file=val_csv, data_dir='data/raw/training_png', batch_size=args.batch_size, is_train=False
    # )
    train_dataloader = create_tile_dataloader(src_dir=train_dir, batch_size=args.batch_size, is_train=True)
    val_dataloader = create_tile_dataloader(src_dir=val_dir, batch_size=args.batch_size, is_train=False)
    # Model
    logger.info("Constructing model")
    model = LitResnet(num_classes=1, lr=args.lr)

    # Callbacks
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    checkpoint_dir = os.path.join('data/outputs', timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoints_cb = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        filename='tupac-{epoch:02d}-{val_loss:.2f}',
        save_top_k=2,
        mode='min',
        save_last=True,
    )
    # best_ckpt_cb = ModelCheckpoint(
    #     monitor='val_loss',dirpath=checkpoint_dir,filename='best', save_top_k=1, mode='min',
    # )
    early_stopping_cb = EarlyStopping(monitor='val_loss',patience=5, mode='min')

    # Logger
    tb_logger = TensorBoardLogger(checkpoint_dir, name="dig_patho")# , prefix=timestamp)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoints_cb, early_stopping_cb],
        logger=tb_logger,
        # To use with the infinite dataloader
        # limit_train_batches=100, limit_val_batches=10,

        # gpus=1 if torch.cuda.is_available() else 0,  accelerator='dp' if torch.cuda.is_available() else None
    )

    # Train
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    logger = logging.getLogger("Trainer")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(levelname)s | %(asctime)s:: %(filename)s#%(lineno)d : %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)

    parser = argparse.ArgumentParser(description='Train ResNet model on WSI dataset')
    parser.add_argument('--train_path', type=str, default='tupac16/train', help='dir with pickle of train')
    parser.add_argument('--val_path', type=str, default='tupac16/val', help='dir with pickle of val')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum number of epochs')

    args = parser.parse_args()
    logger.info(f"Parsed arguments: {args.__dict__} -> calling main()")
    main(args)
