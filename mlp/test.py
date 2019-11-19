import torch
import numpy as np
import math
from mlp import MLP
from slp import SLP

def test(model_name, model, test_data, test_loader):

  model.load_state_dict(torch.load('./models/' + model_name + '.pt'))

  # configures the model to 'evaluation' mode, which ensures that steps are not logged
  model.eval()

  # checker code from https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627
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
      if(true_label == pred_label):
        correct_count += 1
      all_count += 1

  print("\nModel Accuracy =", (correct_count/all_count))
  return (correct_count/all_count)
