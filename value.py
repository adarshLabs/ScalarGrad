
from collections import defaultdict
import math
class Value:

    def __init__(self, data, _parent=(), _op=''):
        self.data = data
        self.grad = 0
        self._prev = set(_parent)
        self._op = _op
        self._backward = lambda : None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), _op='+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), _op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * (other.data**(-1)), (self, other), _op='/')

        def _backward():
            self.grad += (out.grad * (other.data**(-1)))
            other.grad += - (out.grad * self.data * (other.data**(-2)))

        out._backward = _backward
        return out
    
    def __div__(self, other):
        return self.__truediv__(other)
    
    def __rtruediv__(self, other):
        """Handle scalar / Value (e.g., 6 / x)"""
        other = Value(other)
        return other.__truediv__(self)
    
    def __radd__(self, other):
        return self + other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only scalar exponents supported"
        out = Value(self.data ** other, (self,), _op='**')
        
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        
        out._backward = _backward
        return out
        
    def __relu__(self):
        out = Value(self.data if self.data > 0 else 0, (self,), _op='relu')

        def _backward():
            self.grad += (1 if self.data > 0 else 0) * out.grad

        out._backward = _backward
        return out

    def __tanh__(self):
        out = Value(math.tanh(self.data), (self,), _op='tanh')

        def _backward():
            self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward
        return out

    def backward(self)-> None:
        nodes= []
        visited = set()
        self.grad = 1
        def dfs(curr):
            if curr in visited:
                return
            visited.add(curr)
            for parent in curr._prev:
                dfs(parent)
            nodes.append(curr)
        dfs(self)
        for node in reversed(nodes):
            node._backward()


    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    

        


    
