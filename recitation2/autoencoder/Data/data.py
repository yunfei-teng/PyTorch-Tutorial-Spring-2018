# PyTorch tutorial codes for course EL-9133 Advanced Machine Learning, NYU, Spring 2018
# Data/data.py: load data and process data
# read: http://pytorch.org/docs/master/data.html
# data loaders are *iterators*!
import torch
from torchvision import datasets, transforms
from Pipeline.option import args

# load trainig data loader
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.STL10('./Data/stl10', split='train', download=True,
                   transform=transforms.Compose([
                       transforms.Resize(128), 
                       transforms.RandomHorizontalFlip(),
                       transforms.RandomVerticalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

# load testing data loader
test_loader = torch.utils.data.DataLoader(
    datasets.STL10('./Data/stl10', split='test', 
                   transform=transforms.Compose([
                       transforms.Resize(128),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)