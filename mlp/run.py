import numpy as np
import torchvision
import torch
from mlp import MLP
from slp import SLP
from train import train

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))])

train_data = torchvision.datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

model = MLP(784, 10, [300])
print('training for 3 epochs')
train('MLP_1_NLL_SGD_3', model, torch.nn.NLLLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 3, transform, train_data, train_loader)
print('training for 10 epochs')
train('MLP_1_NLL_SGD_10', model, torch.nn.NLLLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 10, transform, train_data, train_loader)
print('training for 30 epochs')
train('MLP_1_NLL_SGD_30', model, torch.nn.NLLLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 30, transform, train_data, train_loader)

model = MLP(784, 10, [400, 200])
print('training for 3 epochs')
train('MLP_2_NLL_SGD_3', model, torch.nn.NLLLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 3, transform, train_data, train_loader)
print('training for 10 epochs')
train('MLP_2_NLL_SGD_10', model, torch.nn.NLLLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 10, transform, train_data, train_loader)
print('training for 30 epochs')
train('MLP_2_NLL_SGD_30', model, torch.nn.NLLLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 30, transform, train_data, train_loader)

model = MLP(784, 10, [600, 500, 400, 300, 200, 100])
print('training for 3 epochs')
train('MLP_6_NLL_SGD_3', model, torch.nn.NLLLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 3, transform, train_data, train_loader)
print('training for 10 epochs')
train('MLP_6_NLL_SGD_10', model, torch.nn.NLLLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 10, transform, train_data, train_loader)
print('training for 30 epochs')
train('MLP_6_NLL_SGD_30', model, torch.nn.NLLLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 30, transform, train_data, train_loader)


