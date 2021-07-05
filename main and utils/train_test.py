
import torch.optim as optim
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


train_losses = []
train_acc = []
test_losses = []
test_acc = []


def create_optim(model, epochs, trainloader):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.5, steps_per_epoch=len(trainloader), epochs=epochs)
    return optimizer, scheduler


def train_model(epochs, model, trainloader, testloader, optimizer,scheduler,device='cuda'):
    def train(model, device, trainloader, optimizer, epochs):
    model.train()
    print(len(trainloader))
    pbar = tqdm(trainloader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = F.cross_entropy(y_pred, target)
        train_losses.append(loss)
        loss.backward()
        optimizer.step()
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)

    def test(model, device, testloader):    
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(testloader.dataset)
        test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(testloader.dataset),
            100. * correct / len(testloader.dataset)))
        
        test_acc.append(100. * correct / len(testloader.dataset))
    
    for epoch in range(epochs):
        print("EPOCH:", epoch)
        train(model, device, trainloader, optimizer, epochs)
        scheduler.step()
        test(model, device, testloader)
