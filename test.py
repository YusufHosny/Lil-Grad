# inputs
a = [[1, 2], [1, 3]]
b = [[2, 3, 4], [5, 3, 6]]


# Lil grad path
from engine import Tensor
a1 = Tensor(a)
b1 = Tensor(b)
m1 = Tensor(2.)

c1 = a1 @ b1 + m1
n1 = (c1 * 0.5)
d1 = n1.mean()

d1.backward()

# torch verification path
import torch

a2 = torch.tensor(a,dtype=torch.float32)
b2 = torch.tensor(b,dtype=torch.float32)
m2 = torch.tensor(2.)

m2.requires_grad = True
a2.requires_grad = True
b2.requires_grad = True

c2 = a2 @ b2 + m2
n2 = (c2 * 0.5)
d2 = n2.mean()

d2.backward()

print(((a2.grad - a1.grad).max() < 0.0001).item())
print(((b2.grad - b1.grad).max() < 0.0001).item())
print(((m2.grad - m1.grad).max() < 0.0001).item())


print(((a2.data - a1.data).max() < 0.0001).item())
print(((b2.data - b1.data).max() < 0.0001).item())
print(((m2.data - m1.data).max() < 0.0001).item())