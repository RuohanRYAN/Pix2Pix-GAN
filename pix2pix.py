from __future__ import print_function
import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image,make_grid
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

from network import define_G, define_D, GANLoss
from data import get_training_set, get_test_set
from utils import display_image
from utils import save_img,VisdomLinePlotter

global plotter
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
# parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--dataset', type=str,default="facades", help='name of the dataset')

print("=======> load dataset")
opt = parser.parse_args()
root_path = "./datasets/"
train_dataset = get_training_set(root_path+opt.dataset, opt.direction)
test_dataset = get_test_set(root_path+opt.dataset, opt.direction)
train_loader = DataLoader(dataset=train_dataset,num_workers=opt.threads,batch_size=opt.batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_dataset,num_workers=opt.threads,batch_size=opt.batch_size,shuffle=True)


device = torch.device("cuda:0" if opt.cuda else "cpu")
net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic', gpu_id=device)
criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)

optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

plotter = VisdomLinePlotter(env_name="metrics")
#### visualize dataset #########

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    loss_tot_d = []
    loss_tot_g = []
    for i,batch in enumerate(train_loader):
        # print(batch[0].shape)
        # print(batch[0])
        # display_image(batch[0])
        # plt.show()
        # display_image(batch[1])
        # plt.show()
        real_a, real_b = batch[0].to(device), batch[1].to(device)
        fake_b = net_g(real_a)

        optimizer_d.zero_grad()
        fake_ab = torch.cat((real_a,fake_b),dim=1)
        pred_fake = net_d(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        real_ab = torch.cat((real_a,real_b),dim=1)
        pred_real = net_d(real_ab.detach())
        loss_d_real = criterionGAN(pred_real, True)
        loss_d = (loss_d_fake+loss_d_real)*0.5

        loss_d.backward()
        loss_tot_d.append(loss_d.detach().data)
        optimizer_d.step()

        optimizer_g.zero_grad()
        fake_ab = torch.cat((real_a,fake_b),dim=1)
        pred_fake = net_d(fake_ab.detach())
        loss_g_gan = criterionGAN(pred_fake,True)

        loss_g_l1 = criterionL1(fake_b,real_b)*opt.lamb
        loss_g = loss_g_gan+loss_g_l1
        loss_g.backward()
        loss_tot_g.append(loss_g.detach().data)
        optimizer_g.step()
        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
            epoch, i, len(train_loader), loss_d.item(), loss_g.item()))
        # print(fake_b.shape)
        plotter.plot('loss', 'd_loss', 'GAN Loss', i, loss_d.detach().data)
        plotter.plot('loss', 'g_loss', 'GAN Loss', i, loss_g.detach().data)
        if(i%10==0):
            # save_image(make_grid(fake_b.detach()),".\\images\\fake_{}.png".format(i))
            images = fake_b.detach()
            for j in range(images.shape[0]):
                save_img(images[j],"./images/fake_epoch{i}_{j}.png".format(i=i,j=j))
