# My First ANN Adventure

In building my first artificial neural network, I wanted to explore the classic MNIST digit data set with a multi-layer perceptron. I evaluated the performances of different MLP models by varying these properties:

1. Number of hidden layers
2. Epoch size
3. Type of loss function
4. Type of optimizing function

I used the [PyTorch](https://pytorch.org/) library to create these networks.

## Network architecture (MLP)

The `MLP` class I built is defined with the parameters of `input_size = 784` - the number of input nodes for a 28x28 image, `output_size = 10` - the size of the prediction space, and `hidden_layer_sizes` - an array of sizes for the hidden layers.

### Hidden layers

I build models with these types of hidden layers:

1. One hidden layer with 300 nodes
2. Two hidden layers with 400, 200 nodes
3. Six hidden layers with 600, 500, 400, 300, 200, 100 nodes
4. Eight hidden layers all with 64 nodes

### Activation functions

For each hidden layer, a non-linear ReLU activation function is applied in order to make this a multi-layer perceptron.

After the final layer of size `output_size`, we apply either the [logarithmic Softmax](https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#LogSoftmax) classification function or just the [Softmax](https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#Softmax) classification function, both of which maps the values of the 10 predictions to the most likely one.

```python
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
```

## Network architecture (SLP)

I have the intuition that the SLP would not have the capacity to train over this type of data set because the classifying space is not linearly separable. Nonetheless, I wanted to observe the performance.

The architecture for the single-layer perceptron is much simpler. We only use one layer because it does not make a difference how many layers since there is no non-linearily between them.

```python
class SLP(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super(SLP, self).__init__()
        
        # we have a final log(Softmax) function to find the most likely prediction in the last layer
        self.layers = torch.nn.Sequential(torch.nn.Linear(input_size, output_size), torch.nn.LogSoftmax(dim = 1))

    def forward(self, x):
        return self.layers(x)
 ```

### Loss function

I either use the [negative log-likelihood loss](https://pytorch.org/docs/stable/nn.html#nllloss) function, as recommended by this [article](https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627) or I use the [cross entropy loss](https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#CrossEntropyLoss) function.

```python
loss_function = torch.nn.NLLLoss()
# or
loss_function = torch.nn.CrossEntropyLoss()
```

### Optimizing function

For the optimizing function, I use the classic [stochastic gradient descent](https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html) algorithm with a learning rate `lr = 0.001`.

```python
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
```

### Training

I built the `train()` function on a variety of epochs.

```python
def train(model_name, model, loss_function, optimizer, epochs, transform, train_data, train_loader):

    # configures the model to 'train' mode, which ensures that all steps are recorded for back propagation
    model.train()

    for epoch in range(epochs):

        print('\t\t' + str(epoch))

        for train_digits, train_labels in train_loader:

            # flatten the training digit image
            train_digits = train_digits.view(train_digits.shape[0], -1)

            # reset the optimizing gradient to zero
            optimizer.zero_grad()

            # feed forward propagation
            pred = model(train_digits)

            # calculate loss function
            loss = loss_function(pred.squeeze(), train_labels)

            # back-propagate
            loss.backward()

            # change weights based on loss function
            optimizer.step()

    torch.save(model.state_dict(), './models/' + model_name + '.pt')
```

For every epoch, we load up the training data in batches of 64, flatten the image, reset the gradient, and then propagate the data forward. We then calculate the loss and then propagate this back. Finally, we edit the weights based on the optimizing function. Finally, we save the model.

### Saving models

I saved the various different models with different configurations, all found in the `models/` folder. I used the saving / loading tutorial [here](https://pytorch.org/tutorials/beginner/saving_loading_models.html).

The models are saved in the format of `{MLP or SLP}_{# of hidden layers}_{loss function}_{optimizer}_{epochs}.pt`.

### Performance

I observed the following performances.

![table](https://github.com/ziruihao/ml/blob/master/outcome/table.png "Table")


![graph](https://github.com/ziruihao/ml/blob/master/outcome/graph.png "Graph")

It was remarkable that the fewer the hidden layers, the better the performance. This could very much be due to the [overfitting hypothesis](https://stats.stackexchange.com/questions/338255/what-is-effect-of-increasing-number-of-hidden-layers-in-a-feed-forward-nn). The performance with even two hidden layers was still just below that of using just one hidden layer. It seems that for this classification problem, the optimal is one hidden layer.

I also found it surprising that with a single-layer perceptron, we achieved performances that are comparable to using a MLP. Moreover, the performance is most comparable to the one hidden layer models.

Another interesting trend were that the cross entropy loss MLPs experienced the greatest improvements as we increased the epochs from 3 to 30. The negative log-likelihood loss counterparts did not experience as great of an improvement through the epochs.

I wanted to explore this growth further, so I created 20 more models with epochs at 5 intervals from 1 to 100. Here are the results. This was done on a MLP with only 3 hidden layers, each with size 64.

![graph2](https://github.com/ziruihao/ml/blob/master/outcome/graph2.png "Graph2")
