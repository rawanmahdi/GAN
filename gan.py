import generator as gen
import discriminator as disc
import mnist_data as data

import os

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt

import pytorch_lightning as pl


random_seed = 42
torch.manual_seed(random_seed)

BATCH_SIZE=128
AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS=int(os.cpu_count() / 2)



class GAN(pl.LightningModule):
    def __init__(self, latent_dim=100, lr=0.002):
        super().__init__()
        self.save_hyperparameters() 

        self.generator = gen.Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = disc.Discriminator()

        # noise 
        self.validation_z = torch.randn(6, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y) 
    
    def training_step(self, batch, batch_idx, optimizer_idx):     
        # opt1, opt2 = self.optimizers()
        
        real_imgs, _ = batch

        # sample noise data
        z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(real_imgs)    

        # traing generator: maxiize log(D(G(z)))
        if optimizer_idx == 0:
            # fake images training 
            fake_imgs = self(z)
            y_hat = self.discriminator(fake_imgs)
            # real images 
            y = torch.ones(real_imgs.size(0), 1)
            y = y.type_as(fake_imgs)
            
            g_loss = self.adversarial_loss(y_hat, y)
            
            log_dict = {"g_loss": g_loss}

            return {"loss": g_loss, "progress_bar": log_dict, "log": log_dict}
        
        # training discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        if optimizer_idx == 1:
            # real image training: how well does it distinguish real images
            y_hat_real = self.discriminator(real_imgs)
            y_real = torch.ones(real_imgs.size(0), 1)
            y_real = y_real.type_as(real_imgs)

            real_loss = self.adversarial_loss(y_hat_real, y_real)
            
            # fake image training: how well does it distinguish fake images

            fake_imgs = self(z)
            y_hat_fake = self.discriminator(fake_imgs.detach())
            
            y_fake = torch.zeros(real_imgs.size(0), 1) # zeros gives associated formula
            y_fake = y_fake.type_as(real_imgs)

            fake_loss = self.adversarial_loss(y_hat_fake, y_fake)
            
            d_loss = (real_loss + fake_loss) / 2
            log_dict = {"d_loss": d_loss}
            return {"loss": d_loss, "progress_bar": log_dict, "log": log_dict}

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)    

        return [opt_g, opt_d], []

    def plot_imgs(self):
        z = self.validation_z
        z = z.type_as(self.generator.lin1.weight)
        sample_imgs = self(z).cpu()
        
        print('epoch:', self.current_epoch) 
        fig = plt.figure()
        for i in range(sample_imgs.size(0)):
            plt.subplot(2, 3, i+1)
            plt.tight_layout()
            plt.imshow(sample_imgs.detach()[i, 0, :, :], cmap='gray_r', interpolation='none')
            plt.title("Generated Image")
            plt.xticks([])
            plt.yticks([])  
            plt.axis('off')
            plt.show()


    def on_epoch_end(self):
        self.plot_imgs()   

                             