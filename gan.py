import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torchvision.datasets as datasets

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

# defining hyperparams
n_epochs=200
batch_size=64
learning_rate=0.0002
b1=0.5 # decay of first order momentum of gradient
b2 = 0.999 # decay of second order momentum of gradient
n_cpu = 8
latent_dim=100 # latent space (for generator)
sample_interval=400 
img_size = 28
img_channel = 1

img_shape = (1, 28, 28) 

cuda = True if torch.cuda.is_available() else False


#Generator - multiple hidden Linear layers with Leaky ReLu's and Tanh in final layer  
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim,128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128,256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256,512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512,1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


# Discriminator - hidden linear layers with Leaky ReLu's and sigmoind in the final layer
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


# Binary Cross Entropy loss function
adversarial_loss = torch.nn.BCELoss()

generator = Generator()
discriminator = Discriminator()


if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Data
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)

# Optimizers for generator and discriminator 
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



# Training 
for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # Train Generator
        
        optimizer_G.zero_grad()

        # z - noise for generator
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

        # Generate a batch of images for discriminator 
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        #  Train Discriminator

        optimizer_D.zero_grad()

        # d_loss measures how good Discriminator is at distinguishing generator made images from real ones 

        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )
        # save outputs of generator 
        batches_done = epoch * len(dataloader) + i
        if batches_done % sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
