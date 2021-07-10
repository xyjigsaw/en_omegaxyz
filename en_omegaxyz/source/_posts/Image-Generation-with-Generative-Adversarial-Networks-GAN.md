---
title: Image Generation with Generative Adversarial Networks (GAN)
date: 2021-07-09 21:54:51
tags: [Python, machine learning]
categories: technology
math: true
index_img: https://gitee.com/omegaxyz/img/raw/master/upload/dcgan202107092251.jpeg
---


## Introduction

In this article, I implemented some variants of DCGAN[1], which is one of the generative adversarial networks. The DCGAN model has replaced the fully connected layer with the global pooling layer. It is universally acknowledged that the purpose of GAN is to achieve Nash equilibrium between the discriminator and the generator. That is to say, neither of these two models should perform very well. When I applied the DCGAN to dataset CUB200-2011, I find that the loss of the generator converges to zero rapidly while the loss of the discriminator stays high, which shows that the discriminator can not distinguish the fake images from all images in this case. To improve its performance, I add a fully connected layer at the end of the convolution layer in the discriminator. The outcome is better than the original DCGAN. 

## Codes

main.py


```python
# Name: DCGAN_main
# Author: Reacubeth
# Time: 2021/5/28 19:47
# Mail: noverfitting@gmail.com
# Site: www.omegaxyz.com
# *_*coding:utf-8 *_*

from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from DCGAN import Discriminator, Generator


manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

path = 'bird/CUB_200_2011/'
ROOT_TRAIN = path + 'dataset/train/'


workers = 2
batch_size = 128
image_size = 128
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 200
lr = 0.0002  # 0.0002
beta1 = 0.5
ngpu = 4

dataset = dset.ImageFolder(root=ROOT_TRAIN,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)


device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


netG = Generator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
netG.apply(weights_init)
print(netG)

netD = Discriminator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
netD.apply(weights_init)
print(netD)


criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

real_label = 1.
fake_label = 0.

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()


        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

torch.save(netD, 'checkpoint/netD' + str(num_epochs) + '.pth')

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('gan_losses.pdf', bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
plt.savefig('fake_img.pdf', bbox_inches='tight')
HTML(ani.to_jshtml())


real_batch = next(iter(dataloader))

plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))
plt.savefig('real_img.pdf', bbox_inches='tight')


plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.savefig('cmp_img.pdf', bbox_inches='tight')
plt.show()

```


DCGAN.py

```python
# Name: DCGAN.py
# Author: Reacubeth
# Time: 2021/5/28 18:42
# Mail: noverfitting@gmail.com
# Site: www.omegaxyz.com
# *_*coding:utf-8 *_*

import torch.nn as nn
import torch.nn.functional as F

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 128

# Size of feature maps in discriminator
ndf = 128

# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, x):
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.use_fully = True
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8

            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4

            # use_fully
            # nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

        if self.use_fully:
            print('use_fully')
            self.out = nn.Sequential(
                nn.Linear(ndf * 16 * 4 * 4, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        x = self.main(x)
        if self.use_fully:
            x = x.view(x.size(0), -1)
            x = self.out(x)
        return x


```


## Results

The generated images compared to the original images are shown as follows. The left grid contains the real ones, the right images are fake.

![](https://gitee.com/omegaxyz/img/raw/master/upload/cmp_dcgan202107092251.jpg)

It can be observed that the fake images are close to the real images, and even some of the generated images are difficult for people to distinguish from real ones. Of course, some images are slightly blurred with green backgrounds.

In order to see the evolution of the generated images, I also visualize these fake images at different epochs. It is interesting to see that some latent features have been extracted at the early stage. At the 50th iteration, the whole birds' outlines have begun to appear.

![](https://gitee.com/omegaxyz/img/raw/master/upload/epoch_gan202107092236.jpg)

The following figure shows the change of loss during the training process. We can see that the loss of the generator is a little higher than that of the discriminator. One of the possible reasons is the existence of the fully connected layer. We can also see the losses are not steady for both of them, which may lead to the mode collapse.

![](https://gitee.com/omegaxyz/img/raw/master/upload/gan_loss202107092212.jpg)


## Feature Visualization

In this part, I visualize the attention of different intermediate layers of discriminator in DCGAN with the help of Grad-CAM[2]. Grad-CAM uses the gradient information of the last convolution layer flowing into CNN to assign important values to each neuron. Specifically, the gradient of class $$c$$ is calculated by using $$y^c$$(Logits before softmax), and the activation value of the feature graph is defined as $$a^k$$. These backflow gradients are applied with the global pooling strategy on the width and height dimensions (indexed by $$i$$ and $$j$$, respectively) to obtain neuron importance Iights $$\alpha^c_k$$. Equation 1 and 2 describes the process.

$$\alpha_{k}^{c}=\frac{1}{Z} \sum_{i \in w} \sum_{j \in h} \frac{\partial y^{c}}{\partial A_{i j}^{k}}$$

$$L_{G r a d-C A M}^{c}=\operatorname{ReLU}\left(\alpha_{k}^{c} * A^{k}\right)$$


The heatmap of the last convolutional layer of discriminator in DCGAN in the case of dataset CUB200-2011 is shown as follows. The second and third columns are heatmaps of Grad-CAM and Grad-CAM++[3] respectively. The last two columns are the results superimposed on the original images. I choose to visualize the attention that makes the discriminator judge fake. As can be seen from the results, the place with higher heat value is not birds, but the environment in most cases, which valids that the discriminator should not perform so ill in GAN model.


![](https://gitee.com/omegaxyz/img/raw/master/upload/heatmap_gan202107092244.jpg)



[1] Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).

[2] Selvaraju, Ramprasaath R., et al. "Grad-cam: Visual explanations from deep networks via gradient-based localization." Proceedings of the IEEE international conference on computer vision. 2017.

[3] Chattopadhay, Aditya, et al. "Grad-cam++: Generalized gradient-based visual explanations for deep convolutional networks." 2018 IEEE Winter Conference on Applications of Computer Vision (WACV). IEEE, 2018.

OmegaXYZ.com
All rights reserved.
