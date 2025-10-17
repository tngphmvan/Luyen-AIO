from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torch.utils.data import random_split, DataLoader, Subset
from torchvision.datasets import ImageFolder
from pytorch_lightning import seed_everything
seed_everything(42, workers=8, verbose=False)

dataset = ImageFolder(r"D:\Luyen-AIO\data",
                      transform=EfficientNet_B3_Weights.IMAGENET1K_V1.transforms())
# train_size = int(0.9*len(dataset))
# val_size =
train_dataset, val_dataset = random_split(
    dataset=dataset, lengths=[int(0.9*len(dataset)), len(dataset) - int(0.9*len(dataset))])

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=8,
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=8
)
