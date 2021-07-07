''' Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from models import *
from utils import progress_bar
from tqdm import tqdm
from torchsummary import summary
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensor


def generate_dataset():
    transform_train = A.Compose([
        A.Rotate(+5,-5),
        A.RandomCrop(32,32,p=0.1),
        A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=4, min_width=4, fill_value=(0.4914, 0.4822, 0.4465)), 
        A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ToTensor(),
    ])

    transform_test = A.Compose([
        A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    
    return trainloader, testloader


def generate_model(model, input_size = (3,32,32)):
    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    if cuda:
        torch.cuda.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")
    model_generated = model.to(device)
    print(summary(model_generated, input_size=input_size))
    return model_generated

