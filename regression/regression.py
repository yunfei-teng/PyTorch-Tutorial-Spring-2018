# Introduction experiemnt of course EL-9133 Advanced Machine Learning, NYU, Spring 2018
from __future__ import print_function
from itertools import count

import torch
import torch.autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#---------- Generate Target Polinomial Function ----------#
# A polinomial function: y = w_1* x^1 + b_1 + w_2* x^2 + b_2 + ... + w_n* x^n + b_n
# Learnable parameters: w_1, w_2, ..., w_n, b_1, b_2, ..., b_n
# n = POLY_DEGREE
POLY_DEGREE = 4
W_target = torch.randn(POLY_DEGREE, 1) * 5
b_target = torch.randn(1) * 5

#---------- Define Auxiliary Functions ----------#
def make_features(x):
    '''Builds features i.e. a matrix with columns [x, x^2, x^3, x^4].'''
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)

def f(x):
    '''Approximated function.'''
    return x.mm(W_target) + b_target[0]

def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, len(W) - i)
    result += '{:+.2f}'.format(b[0])
    return result

def get_batch(batch_size=32):
    """Builds a batch i.e. (x, f(x)) pair."""
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    return Variable(x), Variable(y)


#---------- Define Model ----------#
model = torch.nn.Linear(W_target.size(0), 1)
optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum=0.0)

#---------- Train ----------#
for batch_idx in count(1):

    # Generate training data
    batch_x, batch_y = get_batch()              

    # Forward pass
    optimizer.zero_grad()
    output = model(batch_x)                     
    loss = F.smooth_l1_loss(output, batch_y)  
    loss_data = loss.data[0]

    # Backward pass
    loss.backward() 
    optimizer.step() 

    # Stop iteration
    if loss_data < 1e-3:  
        break

#---------- Print ----------#
print('Loss: {:.6f} after {} batches'.format(loss_data, batch_idx))
print('==> Learned function:\t' + poly_desc(model.weight.data.view(-1), model.bias.data))
print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))