<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full"></script>
# My First MNIST Adventure

In building my first artificial neural network, I wanted to explore the classic MNIST digit data set with a multi-layer perceptron and support vector machine.

## Multi-layer Perceptron

I used the https://pytorch.org/ library to configure a multi-layer perceptron network.

### Network architecture

The `MLP` class is defined with the parameters of `input_size = 784` - the number of input nodes for a 28x28 image, `output_size = 10` - the size of the prediction space, and `hidden_layer_sizes` - an array of sizes for the hidden layers.

For each hidden layer, a non-linear ReLU activation function is applied in order to make this a multi-layer perceptron.

After the final layer of size `output_size`, we apply PyTorch's $Softmax$ classification function that maps the values of the 10 predictions to the most likely one.

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


