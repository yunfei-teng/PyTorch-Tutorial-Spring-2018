# PyTorch tutorial codes for course EL-9133 Advanced Machine Learning, NYU, Spring 2018
# Pipeline/run.py: define functions train(epoch) and test()
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.utils as utils

from Pipeline.option import args
from Data.data import train_loader, test_loader
from Architecture.model import netG, netD, nz
from Architecture.optim import optimizerG, optimizerD

def train(epoch):
    input = torch.FloatTensor(args.batch_size, 1, 64, 64)
    noise = torch.FloatTensor(args.batch_size, nz, 1, 1)
    label = torch.FloatTensor(args.batch_size)
    fixed_noise = torch.FloatTensor(args.batch_size, nz, 1, 1).normal_(0, 1)
    real_label = 1
    fake_label = 0
    criterion = torch.nn.BCELoss()
    if args.cuda:
        input, label = input.cuda(), label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
        criterion.cuda()

    fixed_noise = Variable(fixed_noise)
    for batch_idx, (data, target) in enumerate(train_loader):
        netD.zero_grad()
        real_cpu = data
        batch_size = real_cpu.size(0)
        if args.cuda:
            real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)

        # train with real
        output = netD(inputv)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        labelv = Variable(label.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        netG.zero_grad()
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, args.epochs, batch_idx, len(train_loader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))

    if epoch % args.save_model_epoch == 0:
        utils.save_image(input,'real_samples.png', normalize=True)
        fake = netG(fixed_noise)
        utils.save_image(fake.data,'fake_samples.png',normalize=True)

        torch.save(netG.state_dict(), 'netG.pth')
        torch.save(netD.state_dict(), 'netD.pth')