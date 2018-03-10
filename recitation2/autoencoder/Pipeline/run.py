# PyTorch tutorial codes for course EL-9133 Advanced Machine Learning, NYU, Spring 2018
# Pipeline/run.py: define functions train(epoch) and test()
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.utils as utils

from Pipeline.option import args
from Data.data import train_loader, test_loader
from Architecture.model import model
from Architecture.optim import optimizer 

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, data)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
                
    if epoch % args.save_image_epoch:
        utils.save_image(data.data, 'origin_pictures.png', normalize=True, scale_each=True)
        utils.save_image(output.data,'reconstruct_pictures.png', normalize=True, scale_each=True)

    if epoch % args.save_model_epoch:
        torch.save(model.state_dict(), 'model.pth')

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.mse_loss(output, data).data[0] # sum up batch loss
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))