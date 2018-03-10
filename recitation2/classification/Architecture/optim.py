# PyTorch tutorial codes for course EL-9133 Advanced Machine Learning, NYU, Spring 2018
# Architecture/optim.py: define optimizer
# read: http://pytorch.org/docs/master/optim.html
import torch.optim as optim

from Pipeline.option import args
from Architecture.model import model

optimizer = None
if args.optimizer =='SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.1)
elif args.optimizer =='Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
else:
    raise ValueError('Wrong name of optimizer')
    
print('\n---Training Details---')
print('batch size:',args.batch_size)
print('seed number', args.seed)

print('\n---Optimization Information---')
print('optimizer:', args.optimizer)
print('lr:', args.lr)
