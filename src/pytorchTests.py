import numpy
import torch
import torch.autograd as autograd
from torch.autograd import Variable
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

y = torch.tensor([4., 5., 6], requires_grad=True)
z = x + y
#print(z)

# BUT z knows something extra.
#print(z.grad_fn)

s = z.sum()
s.backward()
#print(x.grad)
#print(y.grad)

lin = nn.Linear(5, 3)  # maps from R^5 to R^3, parameters A, b
# data is 2x5.  A maps from 5 to 3... can we map "data" under A?
data = torch.randn(2, 5)
#print(data)
#print(lin(data))  # yes


data = torch.randn(2, 2)
#print(data)
#print(F.relu(data))

data = torch.randn(10, 2)
#print(data)
#print(F.softmax(data, dim=0))
#print(F.softmax(data, dim=0).sum())  # Sums to 1 because it is a distribution!
#print(F.log_softmax(data, dim=0))  # theres also log_softmax

i = torch.randn(1,9)
#print(i)
#print(i.view(1, 1, -1))


lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

# initialize the hidden state.
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)
    print("LSTM")
    print(out)
    print(hidden)
    
#inputs = torch.cat(inputs).view(len(inputs), 1, -1)
#out, hidden = lstm(inputs, hidden)