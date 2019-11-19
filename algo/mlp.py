import torch
from collections import OrderedDict

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