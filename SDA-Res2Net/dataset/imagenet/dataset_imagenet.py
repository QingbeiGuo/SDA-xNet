#pytorch -0.2.1
#python -3.6.2
#torchvision - 0.1.9

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def train_loader(path, batch_size=256, num_workers=16, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return DataLoaderX(
        datasets.ImageFolder(path,
                             transforms.Compose([                       
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
                                 normalize,
                             ])),
        batch_size=batch_size,                                  
        shuffle=True,                                          
        num_workers=num_workers,                               
        pin_memory=pin_memory)

def test_loader(path, batch_size=256, num_workers=16, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return DataLoaderX(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)
