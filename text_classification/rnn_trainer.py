from pytorch_lightning import LightningModule
from torch.nn import CrossEntropyLoss, Linear, Dropout, Sequential, RNN, Embedding
import torch
import torch.nn as nn
from data_prepare import train_dataloader, val_dataloader, class_weights, tokenizer, num_classes
from sklearn.metrics import confusion_matrix, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
import seaborn as sns
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger, CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
from torch.optim import AdamW


class RNNModel(LightningModule):
    def __init__(self, vocab_size=None, embedding_dim=128, hidden_dim=256, num_layers=2, lr=1e-3, num_labels=None):
        super().__init__()
        self.save_hyperparameters()

        if vocab_size is None:
            vocab_size = tokenizer.vocab_size

        if num_labels is None:
            num_labels = num_classes

        self.embedding = Embedding(
            vocab_size, embedding_dim, padding_idx=tokenizer.pad_token_id)
        self.rnn = RNN(embedding_dim, hidden_dim, num_layers,
                       batch_first=True, dropout=0.3, bidirectional=True)
        self.dropout = Dropout(0.3)
        self.fc = Sequential(
            Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            Dropout(0.3),
            Linear(hidden_dim, num_labels)
        )
        self.criterion = CrossEntropyLoss(weight=class_weights)
        self.lr = lr
        self.val_predicts = []
        self.val_targets = []

    def forward(self, input_ids, attention_mask=None):
        # Embedding
        # [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(input_ids)

        # RNN
        # [batch_size, seq_len, hidden_dim*2]
        rnn_out, hidden = self.rnn(embedded)

        # Use last hidden state
        if attention_mask is not None:
            # Mask padding tokens
            lengths = attention_mask.sum(dim=1)  # [batch_size]
            batch_size, seq_len, hidden_size = rnn_out.size()
            # Get last non-padding output for each sequence
            idx = (
                lengths - 1).unsqueeze(1).unsqueeze(2).expand(batch_size, 1, hidden_size)
            last_output = rnn_out.gather(1, idx).squeeze(
                1)  # [batch_size, hidden_dim*2]
        else:
            last_output = rnn_out[:, -1, :]  # [batch_size, hidden_dim*2]

        # Dropout and classification
        output = self.dropout(last_output)
        logits = self.fc(output)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        self.log('train_loss', loss, prog_bar=True,
                 on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.val_predicts.extend(preds.cpu().numpy())
        self.val_targets.extend(labels.cpu().numpy())
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        if len(self.val_targets) == 0:
            return

        cm = confusion_matrix(self.val_targets, self.val_predicts)
        cr = classification_report(
            self.val_targets, self.val_predicts, zero_division=0, output_dict=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix at Epoch {}'.format(self.current_epoch))

        if hasattr(self.logger, 'experiment'):
            self.logger.experiment.add_figure(
                'Confusion Matrix', fig, self.current_epoch)

        self.log('accuracy', cr['accuracy'], prog_bar=True)
        self.log('macro_f1', cr['macro avg']['f1-score'], prog_bar=True)
        self.log('weighted_f1', cr['weighted avg']['f1-score'], prog_bar=True)
        self.log('macro_precision', cr['macro avg']
                 ['precision'], prog_bar=True)
        self.log('weighted_precision',
                 cr['weighted avg']['precision'], prog_bar=True)
        self.log('macro_recall', cr['macro avg']['recall'], prog_bar=True)
        self.log('weighted_recall',
                 cr['weighted avg']['recall'], prog_bar=True)

        plt.close(fig)
        self.val_predicts = []
        self.val_targets = []

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }


if __name__ == "__main__":
    logger = TensorBoardLogger("tb_logs", name="rnn-text-classification")
    comet_logger = CometLogger(
        api_key="NYszMwP9QgVU7MlrN2AsTj0vA",
        project_name="rnn-text-classification",
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/rnn',
        filename='best-rnn',
        save_last=True,
        mode='min',
    )

    model = RNNModel(lr=1e-3, embedding_dim=128,
                     hidden_dim=256, num_layers=2)  # num_labels tự động từ num_classes
    trainer = Trainer(
        max_epochs=100,
        gpus=1 if torch.cuda.is_available() else 0,
        logger=[logger, comet_logger],
        callbacks=[checkpoint_callback, EarlyStopping(
            monitor='val_loss', patience=10)],
        gradient_clip_algorithm='norm',
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        min_epochs=20
    )
    trainer.fit(model, train_dataloader, val_dataloader)
