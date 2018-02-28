# PyTorch tutorial codes for course EL-9133 Advanced Machine Learning, NYU, Spring 2018
# Architecture/optim.py: define optimizer
# read: http://pytorch.org/docs/master/optim.html
import torch.optim as optim

from Pipeline.option import args
from Architecture.model import model

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

print('\n---Training Details---')
print('batch size:',args.batch_size)
print('seed number', args.seed)

print('\n---Optimization Information---')
print('optimizer: SGD')
print('lr:', args.lr)
print('momentum:', args.momentum)
