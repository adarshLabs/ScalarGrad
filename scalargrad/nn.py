from .value import Value
import random

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
    

class Neuron(Module):
    def __init__(self, nin, nonlinear=True):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(0)
        self.nl = nonlinear
    
    def __call__(self, inputs):
        out = sum((wi * xi for wi , xi in zip (self.weights, inputs)), self.bias)
        return out.relu() if self.nl else out
    
    def parameters(self):
        return self.weights + [self.bias]
    
    def __repr__(self):
        return f"{'ReLU' if self.nl else 'Linear'} Neuron({len(self.weights)})"
    
class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, inputs):
        out = [n(inputs) for n in self.neurons]
        return out[0] if len(out)==1 else out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    
class MLP(Module):
    def __init__(self, nin, nouts):
        all_layers = [nin] + nouts
        self.layers = [Layer(all_layers[i-1], all_layers[i], nonlinear=i!=len(all_layers)-1) for i in range(1, len(all_layers))]

    def __call__(self, inputs):
        out = inputs
        for layer in self.layers:
            out = layer(out)
        return out
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(l) for l in self.layers)}]"