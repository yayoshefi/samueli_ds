import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from torchvision.models import resnet50

class LitResnet(pl.LightningModule):
    def __init__(self, num_classes=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy(task='binary')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.float())
        acc = self.accuracy(torch.sigmoid(y_hat), y.int())
        self.log('train_loss', loss)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.float())
        acc = self.accuracy(torch.sigmoid(y_hat), y.int())
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.float())
        acc = self.accuracy(torch.sigmoid(y_hat), y.int())
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
