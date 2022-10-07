import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import *
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import time
import copy
import os

batch_size = 64
learning_rate = 1e-3

transforms = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root = "/Users/barathkumar/Documents/Research/DP & FL/Dataset/Handwriting/Handwriting-subset/Train", transform = transforms)
test_dataset = datasets.ImageFolder(root = "/Users/barathkumar/Documents/Research/DP & FL/Dataset/Handwriting/Handwriting-subset/Test", transform = transforms)

print("Done!")