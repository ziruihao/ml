# My First MNIST Adventure

In building my first artificial neural network, I wanted to explore the classic MNIST digit data set with a multi-layer perceptron and support vector machine.

## Multi-layer Perceptron

I used the [PyTorch](https://pytorch.org/) library to configure a multi-layer perceptron network.

### Network architecture

The `MLP` class I built is defined with the parameters of `input_size = 784` - the number of input nodes for a 28x28 image, `output_size = 10` - the size of the prediction space, and `hidden_layer_sizes` - an array of sizes for the hidden layers.

For each hidden layer, a non-linear ReLU activation function is applied in order to make this a multi-layer perceptron.

After the final layer of size `output_size`, we apply PyTorch's [logarithmic Softmax](https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#LogSoftmax) classification function that maps the values of the 10 predictions to the most likely one.

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

I use the [negative log-likelihood loss](https://pytorch.org/docs/stable/nn.html#nllloss) function, as recommended by this [article](https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627), as it goes well with the [logarithmic Softmax](https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#LogSoftmax) function we use in the last layer.

```python
loss_function = torch.nn.NLLLoss()
```

For the optimizing function, I use the classic [stochastic gradient descent](https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html) algorithm with a learning rate `lr = 0.001`.

```python
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
```

### Training

Using the PyTorch library, we train with these following steps with an `epochs = 3` or `epochs = 10`:

```python
# configures the model to 'train' mode, which ensures that all steps are recorded for back propagation
model.train()

epochs = 3

for epoch in range(epochs):

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
```

For every epoch, we load up the training data in batches of 64, flatten the image, reset the gradient, and then propagate the data forward. We then calculate the loss and then propagate this back. Finally, we edit the weights based on the optimizing function.

### Saving models

We save various different models with different configurations, alll found in the `models/` folder. I used the saving / loading tutorial [here](https://pytorch.org/tutorials/beginner/saving_loading_models.html).

1. `one_hidden_layer_three_epochs.pt` A MLP with one hidden layer of size 300, trained on 3 epochs
2. `two_hidden_layers_three_epochs.pt` A MLP with two hidden layers, with sizes `[400, 200]`, trained on 3 epochs
3. `six_hidden_layers_three_epochs.pt` A MLP with six hidden layers, with sizes `[600, 500, 400, 300, 200, 100]`, trained on 3 epochs
4. `one_hidden_layer_ten_epochs.pt` A MLP with one hidden layer of size 300, trained on 10 epochs
5. `two_hidden_layers_ten_epochs.pt` A MLP with two hidden layers, with sizes `[400, 200]`, trained on 10 epochs
6. `six_hidden_layers_ten_epochs.pt` A MLP with six hidden layers, with sizes `[600, 500, 400, 300, 200, 100]`, trained on 10 epochs

### Performance

I observed the following performances.

1. `one_hidden_layer_three_epochs.pt` 86.01%
2. `two_hidden_layers_three_epochs.pt` 78.39%
2. `six_hidden_layers_three_epochs.pt` 10.10%
3. `one_hidden_layer_ten_epochs.pt` 89.50%
2. `two_hidden_layers_ten_epochs.pt` 88.95%
4. `six_hidden_layers_ten_epochs.pt` 11.35%

It is worthy to note that for the MLP with six hidden layers, the performance significantly decreased. This could very much be due to the [overfitting hypothesis](https://stats.stackexchange.com/questions/338255/what-is-effect-of-increasing-number-of-hidden-layers-in-a-feed-forward-nn). The performance with even two hidden layers was still just below that of using just one hidden layer. It seems that for this classification problem, the optimal is one hidden layer.

## Single-layer Perceptron

I have the intuition that the SLP would not have the capacity to train over this type of data set because the classifying space is not linearly separable. Nonetheless, I wanted to observe the performance.

### Network architecture

