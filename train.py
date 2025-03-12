import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from lit_resnet import LitResnet
from wsi_dataset import create_dataloader as create_wsi_dataloader
from tile_dataset import create_dataloader as create_tile_dataloader
import argparse
import torch
import os
from datetime import datetime
import logging 
logger = logging.getLogger(__name__)

def main(args):
    # Data loaders
    train_csv = os.path.join('data/datasets', args.train_csv)
    val_csv = os.path.join('data/datasets', args.val_csv)
    logger.info(f"Construction DataLoader with train_csv: {train_csv}, val_csv: {val_csv}")
    # train_loader = create_wsi_dataloader(
    #     csv_file=train_csv, data_dir='data/raw/training_png', batch_size=args.batch_size, is_train=True
    # )
    # val_loader = create_wsi_dataloader(
    #     csv_file=val_csv, data_dir='data/raw/training_png', batch_size=args.batch_size, is_train=False
    # )
    train_loader = create_tile_dataloader(
        csv_file=train_csv, batch_size=args.batch_size, is_train=True
    )
    val_loader = create_tile_dataloader(
        csv_file=val_csv, batch_size=args.batch_size, is_train=False
    )
    # Model
    logger.info("Constructing model")
    model = LitResnet(num_classes=1, lr=args.lr)

    # Callbacks
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    checkpoint_dir = os.path.join('data/outputs', timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        filename='resnet-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )

    # Logger
    tb_logger = TensorBoardLogger("data/outputs/tb_logs", name="resnet", prefix=timestamp)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=tb_logger,
        # To use with the iteratable dataloader
        limit_train_batches=100, limit_val_batches=10,
        # gpus=1 if torch.cuda.is_available() else 0,  accelerator='dp' if torch.cuda.is_available() else None
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    logger = logging.getLogger("Trainer")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(levelname)s | %(asctime)s:: %(filename)s#%(lineno)d : %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)

    parser = argparse.ArgumentParser(description='Train ResNet model on WSI dataset')
    parser.add_argument('--train_csv', type=str, default='tupac_16_100/data.csv', help='CSV file with training labels')
    parser.add_argument('--val_csv', type=str, default='tupac_16_100/data.csv', help='CSV file with validation labels')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum number of epochs')

    args = parser.parse_args()
    logger.info(f"Parsed arguments: {args.__dict__} -> calling main()")
    main(args)
