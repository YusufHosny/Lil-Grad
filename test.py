# inputs
a = [[1., 2.], [2., 7.], [2., 5.], [8., 4.], [1., 9.]] # 5 x 2
b = [[2., 3., 4.], [5, 3, 6]] # 2 x 3
f = [[1., 2., 8., 4., 6], [2., 6., 8., 5., 6.], [7., 3., 6., 7., 9.], [2., 5., 8., 5., 7.]] # 4 x 5


# Lil grad path
from engine import Tensor
a1 = Tensor(a)
b1 = Tensor(b)
m1 = Tensor(2.)

c1 = a1 @ b1 + m1
d1 = f @ c1
n1 = (d1 * 0.5)

out1 = n1.mean()

out1.backward()

# torch verification path
import torch

a2 = torch.tensor(a,dtype=torch.float32)
b2 = torch.tensor(b,dtype=torch.float32)
m2 = torch.tensor(2.)

m2.requires_grad = True
a2.requires_grad = True
b2.requires_grad = True

c2 = a2 @ b2 + m2
d2 = torch.tensor(f) @ c2
n2 = (d2 * 0.5)

out2 = n2.mean()

out2.backward()

print(((a2.grad - a1.grad).max() < 0.0001).item())
print(((b2.grad - b1.grad).max() < 0.0001).item())
print(((m2.grad - m1.grad).max() < 0.0001).item())


print(((a2.data - a1.data).max() < 0.0001).item())
print(((b2.data - b1.data).max() < 0.0001).item())
print(((m2.data - m1.data).max() < 0.0001).item())