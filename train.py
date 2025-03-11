import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from lit_resnet import LitResnet
from wsi_loader import create_dataloader
import argparse
import torch
import os
from datetime import datetime

def main(args):
    # Data loaders
    train_csv = os.path.join('data/datasets', args.train_csv)
    val_csv = os.path.join('data/datasets', args.val_csv)
    train_loader = create_dataloader(
        csv_file=train_csv, data_dir='data/raw/training_png', batch_size=args.batch_size, is_train=True
    )
    val_loader = create_dataloader(
        csv_file=val_csv, data_dir='data/raw/training_png', batch_size=args.batch_size, is_train=False
    )
    # Model
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
    logger = TensorBoardLogger("data/outputs/tb_logs", name="resnet", prefix=timestamp)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
        # gpus=1 if torch.cuda.is_available() else 0,  accelerator='dp' if torch.cuda.is_available() else None
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet model on WSI dataset')
    parser.add_argument('--train_csv', type=str, default='tupac_16_100/data.csv', help='CSV file with training labels')
    parser.add_argument('--val_csv', type=str, default='tupac_16_100/data.csv', help='CSV file with validation labels')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum number of epochs')

    args = parser.parse_args()
    main(args)
