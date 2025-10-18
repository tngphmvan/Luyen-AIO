from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
from pytorch_lightning import seed_everything

seed_everything(42)
raw = load_dataset(
    'csv',
    # data_files={
    #     "train": r"C:\Users\Vitus\Downloads\UIT-VSFC_train.csv",
    #     "valid": r"C:\Users\Vitus\Downloads\UIT-VSFC_valid.csv",
    #     "test": r"C:\Users\Vitus\Downloads\UIT-VSFC_test.csv"
    # }
    data_files=r"C:\Users\Vitus\Downloads\UIT-VSFC_train.csv"
)

dataset = raw['train'].train_test_split(test_size=0.2, seed=42)

tokenizer = AutoTokenizer.from_pretrained(
    'google-bert/bert-base-multilingual-uncased')


def tokenize_fn(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)


tokenized = dataset.map(tokenize_fn, batched=True,
                        desc="Tokenizing the dataset")

tokenized = tokenized.remove_columns(['text'])
tokenized = tokenized.rename_column('label', 'labels')
tokenized.set_format(
    type='torch',
    # columns=['input_ids', 'attention_mask', 'label']
)
# Đảm bảo nhãn là số nguyên liên tiếp bắt đầu từ 0
# if not isinstance(tokenized['train'].features['labels'], ClassLabel):
tokenized = tokenized.class_encode_column('labels')

# Set format cho PyTorch
tokenized.set_format(type='torch', columns=[
                     'input_ids', 'attention_mask', 'labels'])

# Tính trọng số lớp an toàn (tránh chia cho 0)
# Chuẩn bị WeightedRandomSampler
# Giải thích: train_labels là list các nhãn của tập train có dạng: [0, 1, 0, 2, 1, ...]
train_labels = tokenized['train']['labels']  # list[int]
# Giải thích: labels_tensor là tensor các nhãn của tập train có dtype long
labels_tensor = torch.tensor(train_labels, dtype=torch.long)
# Giải thích: num_classes là số lớp trong bài toán phân loại
num_classes = int(labels_tensor.max().item() + 1)
# Giải thích: class_counts là tensor đếm số lượng mẫu của mỗi lớp
class_counts = torch.bincount(labels_tensor, minlength=num_classes).float()
# Giải thích: class_weights là tensor trọng số của mỗi lớp, tính bằng nghịch đảo của số lượng mẫu, có dạng: [1/count_class_0, 1/count_class_1, ...]
class_counts[class_counts == 0] = 1.0  # tránh chia cho 0
class_weights = 1.0 / class_counts
# Trọng số theo từng mẫu, dtype double cho WeightedRandomSampler
sample_weights = class_weights[labels_tensor].to(dtype=torch.float)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_labels),
    replacement=True
)


train_dataloader = DataLoader(
    tokenized['train'],
    sampler=sampler,
    batch_size=32,
    num_workers=4
)
val_dataloader = DataLoader(
    tokenized['test'],
    batch_size=32,
    shuffle=False,
    num_workers=4
)

# Export cho các trainer khác sử dụng
__all__ = ['train_dataloader', 'val_dataloader',
           'class_weights', 'tokenizer', 'tokenized', 'num_classes']

# In thông tin debug
print(f"Number of classes: {num_classes}")
print(f"Class counts: {class_counts}")
print(f"Class weights: {class_weights}")
print(f"Unique labels in train: {torch.unique(labels_tensor)}")
