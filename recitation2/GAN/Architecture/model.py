# PyTorch tutorial codes for course EL-9133 Advanced Machine Learning, NYU, Spring 2018
# Architecture/model.py: define model
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from Pipeline.option import args

ngf, ndf, nz, nc = 64, 64, 100, 1

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        ngf = 64
        self.main = nn.Sequential(
            nn.ConvTranspose2d( nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 1),
            nn.ReLU(True),
            # (ngf) x 16 x 16
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            #(nc) x 32 x 32
        )
    def forward(self, x):
        output = self.main(x)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x)
        return output.view(-1, 1).squeeze(1)

netG = Generator()
netD = Discriminator()
if args.continue_training:
    netG.load_state_dict(torch.load('netG.pth'))
    netD.load_state_dict(torch.load('netD.pth'))
if args.cuda:
    netG.cuda()
    netD.cuda()
    
print('\n---Model Information---')
print('netG:',netG)
print('netD:',netD)
print('Use GPU:', args.cuda)
