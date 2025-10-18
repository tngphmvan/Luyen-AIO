import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report
import numpy as np
import os
import cv2


class SimpleVideoDataset(Dataset):
    def __init__(self, root, classes, transform=None, frames_per_clip=16):
        self.samples = []
        self.labels = []
        self.classes = classes
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        for label in classes:
            folder = os.path.join(root, label)
            for fname in os.listdir(folder):
                if fname.lower().endswith((".mp4", ".avi", ".mov")):
                    self.samples.append(os.path.join(folder, fname))
                    self.labels.append(self.class_to_idx[label])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.labels[idx]
        cap = cv2.VideoCapture(path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, frame_count-1, self.frames_per_clip, dtype=int)
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((112, 112, 3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        video = torch.stack(frames).permute(1, 0, 2, 3)  # (C, T, H, W)
        return video, label


class VideoCNNClassifier(LightningModule):
    def __init__(self, model_name='r3d_18', num_classes=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        if model_name == 'r3d_18':
            self.model = r3d_18(weights='DEFAULT')
        elif model_name == 'mc3_18':
            self.model = mc3_18(weights='DEFAULT')
        elif model_name == 'r2plus1d_18':
            self.model = r2plus1d_18(weights='DEFAULT')
        else:
            raise ValueError('Unknown model')
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
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
        report = classification_report(
            self.val_targets, self.val_preds, output_dict=True, zero_division=0)
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
    classes = sorted(os.listdir('data/train'))
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.43216, 0.394666, 0.37645], [
                             0.22803, 0.22145, 0.216989])
    ])
    train_dataset = SimpleVideoDataset(
        'data/train', classes, transform=transform)
    val_dataset = SimpleVideoDataset('data/val', classes, transform=transform)
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4,
                            shuffle=False, num_workers=2)
    model = VideoCNNClassifier(
        model_name='r3d_18', num_classes=len(classes), lr=1e-3)
    trainer = Trainer(
        max_epochs=30,
        logger=TensorBoardLogger('tb_logs', name='video-cnn-classification'),
        callbacks=[EarlyStopping(monitor='val_loss', patience=7), ModelCheckpoint(
            monitor='val_loss', save_last=True)],
        accelerator='auto', devices='auto', strategy='auto', gradient_clip_val=1.0
    )
    trainer.fit(model, train_loader, val_loader)
