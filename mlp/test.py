import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
import sys
import math
from mlp import MLP
from slp import SLP

model_name = str(sys.argv[1])

transform = transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))])

test_data = torchvision.datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

# model = MLP(784, 10, [400, 200])
model = SLP(784, 10)

model.load_state_dict(torch.load('./models/' + model_name + '.pt'))

print(model.summary())

# configures the model to 'evaluation' mode, which ensures that steps are not logged
model.eval()

# checker code from https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627
correct_count, all_count = 0, 0
for test_digits,test_labels in test_loader:
  for i in range(len(test_labels)):
    img = test_digits[i].view(1, 784)
    with torch.no_grad():
        logps = model(img)

    
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = test_labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))