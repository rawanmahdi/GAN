#%%
import mnist_data as data
import gan 

import os

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt

import pytorch_lightning as pl


random_seed = 42
torch.manual_seed(random_seed)

BATCH_SIZE=128
AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS=int(os.cpu_count() / 2)
#%%
dm = data.MNISTDataModule()
model = gan.GAN()
#%%
model.plot_imgs()

#%%
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, dm)