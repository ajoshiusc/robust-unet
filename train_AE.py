from __future__ import print_function
import argparse
import h5py
import numpy as np
import os
import time
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import torchvision.utils as vutils
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import math
from sklearn.datasets import make_blobs
from scipy.ndimage import gaussian_filter
from Unet_AE import UNet
pret = 0
random.seed(8)

input_size=64


            
def show_and_save(file_name,img):
    f = "%s.png" % file_name
    save_image(img[2:3,:,:],f)
    
    #fig = plt.figure(dpi=300)
    #fig.suptitle(file_name, fontsize=14, fontweight='bold')
    #plt.imshow(npimg)
    #plt.imsave(f,npimg)
    
def save_model(epoch, AE):
    torch.save(AE.cpu().state_dict(), 'results/AE_%d.pth' % epoch)
    AE.cuda()

  


d=np.load('/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/old results/data__maryland_histeq.npz')
X=d['data']
X=X[0:2380,:,:,:]
X_train=X[0:-20*20,:,:,:]
X_valid=X[-20*20:,:,:,:]


d=np.load('/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/old results/data__TBI_histeq.npz')
X_train=np.concatenate((X_train,d['data'][0:-20*20,:,:,:]))
X_valid=np.concatenate((X_valid,d['data'][-20*20:,:,:,:]))







X_train = np.transpose(X_train[:,::2,::2,:], (0, 3, 1,2))
X_valid = np.transpose(X_valid[:,::2,::2,:] , (0, 3, 1,2))








input = torch.from_numpy(X_train).float()
validation_data = torch.from_numpy(X_valid).float()

batch_size=8


torch.manual_seed(7)
train_loader = torch.utils.data.DataLoader(input,
                                           batch_size=batch_size,
                                           shuffle=True)
Validation_loader = torch.utils.data.DataLoader(validation_data,
                                          batch_size=batch_size,
                                          shuffle=True)
###### define constant########
input_channels = 3
hidden_size =128
max_epochs =30
lr = 3e-4
beta =0

######learning rate scadular#####
def exp_lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=10):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch:
        return optimizer
    
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer


##########load low res net##########
G=UNet(3, 3).cuda()
#load_model(epoch,G.encoder, G.decoder,LM)
opt_enc = optim.Adam(G.parameters(), lr=lr)
data = next(iter(Validation_loader))
fixed_batch = Variable(data).cuda()

#######losss#################




def MSE_loss(Y, X):
    msk = torch.tensor(X > -1e-6).float()
    ret = ((X- Y) ** 2)*msk
    ret = torch.sum(ret,1)
    return ret 
def BMSE_loss(Y, X, beta,sigma,Dim):
    term1 = -((1+beta) / beta)
    K1=1/pow((2*math.pi*( sigma** 2)),(beta*Dim/2))
    term2=MSE_loss(Y, X)
    term3=torch.exp(-(beta/(2*( sigma** 2)))*term2)
    loss1=torch.sum(term1*(K1*term3-1))
    return loss1


# Reconstruction + KL divergence losses summed over all elements and batch
def beta_loss_function(recon_x, x, beta):
    msk = torch.tensor(x > 1e-6).float()

    if beta > 0:
        sigma=1
        # If beta is nonzero, use the beta entropy
        BBCE = BMSE_loss(recon_x.view(-1, 64*64*3), x.view(-1, 64*64*3), beta,sigma,64*64*3)
    else:
        # if beta is zero use binary cross entropy
        BBCE = torch.sum(MSE_loss(recon_x.view(-1, 64*64*3),x.view(-1, 64*64*3)))

    # compute KL divergence
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    
    return BBCE



pay=0
train_loss=0
valid_loss=0
valid_loss_list, train_loss_list= [], []
for epoch in range(max_epochs):
    train_loss=0
    valid_loss=0
    opt_enc=exp_lr_scheduler(opt_enc, epoch, lr_decay=0.1, lr_decay_epoch=10)
    for data in train_loader:
        batch_size = data.size()[0]

        #print (data.size())
        datav = Variable(data).cuda()
        #datav[l2,:,row2:row2+5,:]=0

        rec_enc = G(datav)
        beta_err=beta_loss_function(rec_enc, datav,beta) 
        err_enc = beta_err
        opt_enc.zero_grad()
        err_enc.backward()
        opt_enc.step()
        train_loss+=beta_err.item()
    train_loss /= len(train_loader.dataset)



    G.eval()
    with torch.no_grad():
        for data in Validation_loader:
            data = Variable(data).cuda()
            valid_rec = G(data)
            beta_err=beta_loss_function(valid_rec, data,beta) 
            valid_loss+=beta_err.item()
        valid_loss /= len(Validation_loader.dataset)

    if epoch == 0:
        best_val = valid_loss
    elif (valid_loss < best_val):
        save_model(epoch, G)
        pay=0
        best_val = valid_loss
    pay=pay+1
    if(pay==100):
        break



    
    print(valid_loss)
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    rec_imgs = G(fixed_batch)
    show_and_save('results/Input_epoch_%d.png' % epoch ,make_grid((fixed_batch.data[:,2:3,:,:]).cpu(),8))
    show_and_save('results/rec_epoch_%d.png' % epoch ,make_grid((rec_imgs.data[:,2:3,:,:]).cpu(),8))
    show_and_save('results/Error_epoch_%d.png' % epoch ,make_grid((fixed_batch.data[:,2:3,:,:]-rec_imgs.data[:,2:3,:,:]).cpu(),8))

    #localtime = time.asctime( time.localtime(time.time()) )
    #D_real_list_np=(D_real_list).to('cpu')
save_model(epoch, G.encoder, G.decoder)    
plt.plot(train_loss_list, label="train loss")
plt.plot(valid_loss_list, label="validation loss")
plt.legend()
plt.show()  