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
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
                
    if epoch % args.save_model_epoch:
        torch.save(model.state_dict(), 'model.pth')

    if args.visualize and epoch % args.visualize_epoch == 0:
        output = model.visual_backprop(data)
        utils.save_image(data.data, 'origin_pictures.png', normalize=True, scale_each=True)
        utils.save_image(output.data, 'visual_results.png', normalize=True, scale_each=True)
        
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
