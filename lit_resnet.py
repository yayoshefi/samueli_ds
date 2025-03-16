import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
# from torchvision.models import resnet50
from torchvision import models

class LitResnet(pl.LightningModule):
    def __init__(self, num_classes=1, lr=1e-3, arch="resnet50"):
        super().__init__()
        self.save_hyperparameters()
        self.model = getattr(models, arch)(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy(task='binary' if num_classes==1 else "multiclass")

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch):
        x, y = batch[:2]
        y_hat = self(x)
        loss = self.criterion(y_hat, y.float()) 
        acc = self.accuracy(torch.sigmoid(y_hat), y.int())
        return loss, acc
    
    def training_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch)
        self.log('train_loss', loss, batch_size=len(batch))
        self.log('train_acc', acc, prog_bar=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch)
        self.log('val_loss', loss, prog_bar=True, batch_size=len(batch))
        self.log('val_acc', acc, prog_bar=True, batch_size=len(batch))

    def test_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch)
        self.log('test_loss', loss, batch_size=len(batch))
        self.log('test_acc', acc, batch_size=len(batch))
    
    def predict_step(self, batch, batch_idx):
        x, y, data = batch
        y_hat = self(x)
        y_hat = torch.sigmoid(y_hat)
        return y_hat, data

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        optimizer_cfg = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3),
                "monitor": "val_loss",
                "frequency": 1,
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
            },
        }
        return optimizer_cfg
