import math
import matplotlib
import matplotlib.pyplot as plt

import random
import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op='', label='', dtype='float64'):
        # initialize data into numpy arrays
        if(isinstance(data, np.ndarray)):
            d = data
        elif(isinstance(data, list)):
            d = np.array(data, dtype=dtype)
        else:
            d = np.array([data])
 
        self.data = d
        self.T = self.data.T
        self.shape = self.data.shape
        self.grad = np.zeros_like(d, dtype=dtype)

        # no backward function for leaf nodes
        self._backward = lambda: None

        self._prev = set(_children)
        self.label = label
        self._op = _op

        # wrapper method
    def __repr__(self):
        return f"Tensor({self.data})"

    # basic add operation
    def __add__(self, other):
        # if other is not a Tensor object, instantiate a leaf Tensor oject with other as the Tensor
        other = other if isinstance(other, Tensor) else Tensor(other)
        # create the output Tensor object
        out = Tensor(self.data + other.data, (self, other), '+')

        # define the backward function with the chain rule derivative of an addition
        def _backward():
            if(self.shape == other.shape):
                self.grad += 1.0 * out.grad
                other.grad += 1.0 * out.grad
            elif(self.shape == (1,)):
                self.grad += (1.0 * out.grad).sum()
                other.grad += 1.0 * out.grad
            elif(other.shape == (1,)):
                self.grad += 1.0 * out.grad
                other.grad += (1.0 * out.grad).sum()
            elif(len(self.shape) == 1):
                self.grad += (1.0 * out.grad).sum(0)
                other.grad += 1.0 * out.grad
            elif(len(other.shape) == 1):
                self.grad += 1.0 * out.grad
                other.grad += (1.0 * out.grad).sum(0)
            else:
                raise Exception("Broadcast Error: Only broadcasting from a scalar or single element list is allowed.")


        out._backward = _backward

        return out
    
    # addition of other + self
    def __radd__(self, other):
        return self + other
    

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            if(self.shape == other.shape):
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
            elif(self.shape == (1,)):
                self.grad += (other.data * out.grad).sum()
                other.grad += self.data * out.grad
            elif(other.shape == (1,)):
                self.grad += other.data * out.grad
                other.grad += (self.data * out.grad).sum()
            else:
                raise Exception("Broadcast Error: Only broadcasting from a scalar or single element list is allowed.")

        out._backward = _backward

        return out
    
    def __rmul__(self, other):
        return self * other
    
    def exp(self):
        x = self.data
        out = Tensor(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        
        return out
    
    def __truediv__(self, other):
        return self * (other**(-1))
    
    def __rtruediv__(self, other):
        return other * (self**(-1))


    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting ints and floats"
        out = Tensor(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * (self.data**(other-1)) * out.grad

        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), 'matmul')

        def _backward():
            self.grad += out.grad @ other.T
            other.grad += self.T @ out.grad


        out._backward = _backward
        
        return out
    
    def __rmatmul__(self, other):
        assert isinstance(other, (np.ndarray, Tensor, list)), "only supporting matrix multiplication with numpy arrays, Tensors, and lists"
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(((self.T @ other.T).T), (self, other), 'matmul')
        
        def _backward():
            self.grad += other.T @ out.grad
            other.grad += out.grad @ self.T
        out._backward = _backward

        return out
    
    def mean(self):
        out = Tensor(self.data.mean(), (self,), 'mean')

        def _backward():
            self.grad += (1/self.data.size) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Tensor(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1.0 - (out.data ** 2)) * out.grad

        out._backward = _backward

        return out
    
    
    def relu(self):
        x = self.data
        r = np.zeros_like(x.reshape(-1))
        for i, xi in enumerate(x.reshape(-1)):
            r[i] = xi if xi > 0 else 0

        r = r.reshape(x.shape)

        out = Tensor(r, (self, ), 'ReLU')

        def _backward():
            self.grad += (r/x) * out.grad

        out._backward = _backward

        return out
    
    def __len__(self):
        return len(self.data)
    
    def sum(self):
        out = Tensor(self.data.sum(), (self,), 'summation')

        def _backward():
            self.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    @staticmethod
    def ones(shapelike):
        return Tensor(np.ones(shapelike))
    
    @staticmethod
    def ones_like(like):
        return Tensor(np.ones_like(like))
    
    @staticmethod
    def zeros(shapelike):
        return Tensor(np.zeros(shapelike))
    
    @staticmethod
    def zeros_like(like):
        return Tensor(np.zeros_like(like))
    
    @staticmethod
    def randn(shapelike):
        return Tensor(np.random.default_rng().standard_normal(size=shapelike))
    
    @staticmethod
    def rand(shapelike):
        return Tensor(np.random.default_rng().uniform(low=-1, size=shapelike))
    
    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def __getitem__(self, i):
        out = Tensor(self.data[i], (self,), 'indexing')
        
        def _backward():
            self.grad[i] += 1.0 * out.grad
        out._backward = _backward

        return out

    def __iter__(self):
        self.ix = 0
        return self
    
    def __next__(self):
        if self.ix == len(self.data):
            raise StopIteration
        else:          
            out = self[self.ix]
            self.ix += 1        
            return out
        
    def nelement(self):
        return self.data.size
    
    def flatten(self):
        out = Tensor(self.data.reshape(-1), (self), 'flatten')

        def _backward():
            self.grad = out.grad.reshape(-1)
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        self.grad = np.array([1])
        for node in reversed(topo):
            node._backward()
