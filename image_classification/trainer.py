from pytorch_lightning import seed_everything, LightningModule, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.nn import (CrossEntropyLoss, KLDivLoss, Dropout,
                      Linear, ReLU, Softmax, Sequential)
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from functools import cache
from data_prepare import train_dataloader, val_dataloader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

seed_everything(seed=42, workers=8, verbose=False)


@cache
def _get_model(weight: bool = True):
    model = efficientnet_b3(
        weights=EfficientNet_B3_Weights.IMAGENET1K_V1 if weight else None)

    model.classifier = Sequential(
        Dropout(0.3, inplace=True),
        Linear(1536, 10, bias=True),
        Softmax(dim=1)
    )

    return model


class WrapperModel(LightningModule):
    def __init__(self, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = _get_model()
        self.criterion = CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.val_predictions = []
        self.val_targets = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.val_predictions.append(y_hat.detach().cpu())
        self.val_targets.append(y.detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.val_predictions)
        all_targets = torch.cat(self.val_targets)
        report = classification_report(
            all_targets, np.argmax(all_preds.numpy(), axis=1), output_dict=True, zero_division=0)
        self.log('accuracy', report['accuracy'], prog_bar=True)
        self.log('precision', report['weighted avg']
                 ['precision'], prog_bar=True)
        self.log('recall', report['weighted avg']
                 ['recall'], prog_bar=True)
        self.log('f1_score', report['weighted avg']
                 ['f1-score'], prog_bar=True)
        self.val_predictions.clear()
        self.val_targets.clear()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }


if __name__ == '__main__':
    # Example usage
    model = WrapperModel()
    trainer = Trainer(
        max_epochs=100,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10),
            ModelCheckpoint(dirpath="checkpoints",
                            filename="best", monitor="val_loss", save_last=True)
        ],
        accelerator="auto",
        # accumulate_grad_batches=len(train_dataloader)//64,
        devices="auto",
        strategy="auto",
        gradient_clip_val=1.0,
        logger=TensorBoardLogger("lightning_logs", name="efficientnet_b3")
    )
    trainer.fit(model, train_dataloader, val_dataloader)
