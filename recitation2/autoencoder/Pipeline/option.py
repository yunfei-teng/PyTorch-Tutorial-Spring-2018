# PyTorch tutorial codes for course EL-9133 Advanced Machine Learning, NYU, Spring 2018
# Pipeline/option.py: pptions
from __future__ import print_function
import torch
import argparse

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--optimizer',default='Adam', metavar='OPTM',
                    help='define optimizer (default: Adam)')  
parser.add_argument('--dataset',default='stl10', metavar='DSET',
                    help='define dataset')              
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--continue_training', action='store_true', default=False,
                     help='continue training')
parser.add_argument('--visualize', action='store_true', default=False,
                    help='visual backprop')
parser.add_argument('--use_unet', action='store_true', default=False,
                    help='enables unet')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_image_epoch', type=int, default=5, metavar='N',
                    help='how many epochs to wait before saving image')
parser.add_argument('--save_model_epoch', type=int, default=10, metavar='N',
                    help='how many epochs to wait before saving model')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()