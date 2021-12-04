"""
@kingjuno
Geo
"""

import torch
import torch.nn as nn
from torch.utils import data
import torchvision
from torchvision.transforms import ToTensor

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_mnist(batch_size = 128, device=default_device):
    """
    Loads the MNIST dataset, transforms it and returns a DataLoader.
    """
    transform = ToTensor()
    train_set = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader, train_set, test_set
