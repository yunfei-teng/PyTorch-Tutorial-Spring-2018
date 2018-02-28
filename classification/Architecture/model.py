# PyTorch tutorial codes for course EL-9133 Advanced Machine Learning, NYU, Spring 2018
# Architecture/model.py: define model
import torch.nn as nn
import torch.nn.functional as F

from Pipeline.option import args

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        nf = 8
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, etc.)
        self.conv1 = nn.Conv2d(    1, nf* 1, 5, 1, 0) #24
        self.conv2 = nn.Conv2d(nf* 1, nf* 2, 4, 2, 1) #12
        self.conv3 = nn.Conv2d(nf* 2, nf* 4, 5, 1, 0) #8
        self.conv4 = nn.Conv2d(nf* 4, nf* 8, 4, 2, 1) #4
        self.conv5 = nn.Conv2d(nf* 8,    10, 4, 1, 0) #1
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100,  10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

model = None
if args.linear:
    model = LinearNet()
else:
    model = ConvNet()
if args.cuda:
    model.cuda()
    
print('\n---Model Information---')
print('Net:',model)
print('Use GPU:', args.cuda)
