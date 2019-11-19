import torch

class SLP(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super(SLP, self).__init__()
        
        # we have a final log(Softmax) function to find the most likely prediction in the last layer
        self.layers = torch.nn.Sequential(torch.nn.Linear(input_size, output_size), torch.nn.LogSoftmax(dim = 1))

    def forward(self, x):
        return self.layers(x)