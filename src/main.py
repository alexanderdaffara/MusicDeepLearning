import numpy
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

V_data = [1., 2., 3.]
V = torch.tensor(V_data)
#print(V)

M_data = [[1., 2., 3.], [4., 5., 6]]
M = torch.tensor(M_data)
#print(M)

T_data = [[[1., 2., 3.], [3., 4., 5.]],
          [[5., 6., 7.], [7., 8., 9.]]]
T = torch.tensor(T_data)

#print(T)

#print(V[0].item())
#print(M[0])
#print(T[0])

x = torch.randn((3, 4, 5))
#print(x)

x_1 = torch.randn(2, 3)
y_1 = torch.randn(2, 5)
z_1 = torch.cat([x_1, y_1], 1)
#print(z_1)

x = torch.tensor([1., 2., 3], requires_grad=True)

# With requires_grad=True, you can still do all the operations you previously
# could
y = torch.tensor([4., 5., 6], requires_grad=True)
z = x + y
print(z)

# BUT z knows something extra.
print(z.grad_fn)