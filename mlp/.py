import torch
import numpy as np
import torchvision
from time import time
from torchvision import datasets, transforms
import csv
import math
from collections import OrderedDict

# train_data = []
# train_labels = []

# def load_training_data(path):
#     with open(path, mode='r') as train_input:
#         reader = csv.reader(train_input, delimiter=',', quotechar='"')

#         rows_read = 0

#         for row in reader:
#             if (rows_read is not 0):
#                 train_data.append(list(map(int, row[1:])))
#                 train_labels.append(list(map(int, row[0])))
#             rows_read = rows_read + 1
#     train_input.close()

# load_training_data('./data/train.csv')

# train_data = torch.tensor(train_data, dtype=torch.float)

# train_labels = torch.tensor(train_labels, dtype=torch.float)

transform = transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))])

train_data = torchvision.datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
test_data = torchvision.datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

class MLP(torch.nn.Module):

    def __init__(self, input_size, output_size, hidden_layer_sizes):
        super(MLP, self).__init__()

        # the layers that define the network
        layers = []
        
        prev_layer_size = input_size
        
        for i, hidden_layer_size in enumerate(hidden_layer_sizes):
            layers.append((str(i) + 'linear', torch.nn.Linear(prev_layer_size, hidden_layer_size)))
            # we have a non-linear ReLU activation layer after each linear
            layers.append((str(i) + 'activation', torch.nn.ReLU()))
            prev_layer_size = hidden_layer_size

        layers.append((str(len(hidden_layer_sizes)) + 'linear', torch.nn.Linear(prev_layer_size, output_size)))
        # we have a final log(Softmax) function to find the most likely prediction in the last layer
        layers.append(('final activation', torch.nn.LogSoftmax(dim = 1)))

        self.layers = torch.nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.layers(x)

model = MLP(784, 10, [300])

loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

model.train()

epochs = 3

for epoch in range(epochs):

    print(epoch)

    for train_digits, train_labels in train_loader:

        train_digits = train_digits.view(train_digits.shape[0], -1)

        optimizer.zero_grad()

        pred = model(train_digits)

        loss = loss_function(pred.squeeze(), train_labels)

        loss.backward()
        optimizer.step()

model.eval()

test_digits, test_labels = next(iter(test_loader))
test_digits = test_digits.view(test_digits.shape[0], -1)

correct_count, all_count = 0, 0
for test_digits, test_labels in test_loader:
  for i in range(len(test_labels)):
    img = test_digits[i].view(1, 784)
    with torch.no_grad():
        logps = model(img)
    
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = test_labels.numpy()[i]
    if (true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))