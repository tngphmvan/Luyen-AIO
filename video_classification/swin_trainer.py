import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchvision import transforms
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report
import numpy as np
import os

# You need to install timm for Swin Transformer video models
# pip install timm
import timm

class VideoDataset(DatasetFolder):
    def __init__(self, root, transform=None, extensions=('mp4', 'avi', 'mov')):
        super().__init__(root, loader=self.video_loader, extensions=extensions)
        self.transform = transform
    def video_loader(self, path):
        import decord
        vr = decord.VideoReader(path)
        idxs = np.linspace(0, len(vr)-1, 32, dtype=int)  # Swin thường dùng 32 frames
        frames = vr.get_batch(idxs).asnumpy()
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        frames = torch.stack(frames).permute(1, 0, 2, 3)  # (C, T, H, W)
        return frames

class SwinVideoClassifier(LightningModule):
    def __init__(self, model_name='swin3d_tiny_patch244_window877_kinetics400_1k', num_classes=2, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.val_preds = []
        self.val_targets = []
    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_preds.extend(preds.cpu().numpy())
        self.val_targets.extend(y.cpu().numpy())
        self.log('val_loss', loss, prog_bar=True)
        return loss
    def on_validation_epoch_end(self):
        if not self.val_targets:
            return
        report = classification_report(self.val_targets, self.val_preds, output_dict=True, zero_division=0)
        self.log('accuracy', report['accuracy'], prog_bar=True)
        self.log('macro_f1', report['macro avg']['f1-score'], prog_bar=True)
        self.val_preds.clear()
        self.val_targets.clear()
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}

if __name__ == '__main__':
    seed_everything(42)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = VideoDataset('data/train', transform=transform)
    val_dataset = VideoDataset('data/val', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)
    model = SwinVideoClassifier(num_classes=len(train_dataset.classes), lr=1e-4)
    trainer = Trainer(
        max_epochs=30,
        logger=TensorBoardLogger('tb_logs', name='swin-video-classification'),
        callbacks=[EarlyStopping(monitor='val_loss', patience=7), ModelCheckpoint(monitor='val_loss', save_last=True)],
        accelerator='auto', devices='auto', strategy='auto', gradient_clip_val=1.0
    )
    trainer.fit(model, train_loader, val_loader)
