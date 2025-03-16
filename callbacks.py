import time
import logging
import pytorch_lightning as pl

# Configure Python logger
logger = logging.getLogger("UTILS")

class EpochTimerCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_duration = time.time() - self.epoch_start_time
                # Get batch size and num_workers from DataLoader
        dataloader = trainer.train_dataloader
        num_workers = dataloader.num_workers if hasattr(dataloader, 'num_workers') else "Unknown"
        # Handle batch size when using a Sampler
        batch_size = getattr(dataloader, 'batch_size', None)
        if batch_size is None and hasattr(dataloader, 'dataset'):
            batch_size = getattr(dataloader.dataset, 'batch_size', 'Unknown')

        logger.info(
            f"Epoch {trainer.current_epoch} completed in {epoch_duration:.2f} sec | "
            f"Batch Size: {batch_size} | Num Workers: {num_workers}"
        )
