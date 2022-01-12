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
from AE import UNet
pret = 0
random.seed(4)

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



#d=np.load('/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/old results/data_24_ISEL_histeq.npz')
#X = d['data']
#X_data = X[0:10*20, :, :, 0:3]
#X_valid=np.concatenate((X_valid,X_data))
#X_data = X[10*20:15*20, :, :, 0:3]
#X_valid=np.concatenate((X_valid,X_data))








X_train = np.transpose(X_train[:,::2,::2,:], (0, 3, 1,2))
X_valid = np.transpose(X_valid[:,::2,::2,:] , (0, 3, 1,2))








input = torch.from_numpy(X_train).float()
validation_data = torch.from_numpy(X_valid).float()

batch_size=8


torch.manual_seed(10)
train_loader = torch.utils.data.DataLoader(input,
                                           batch_size=batch_size,
                                           shuffle=True)
Validation_loader = torch.utils.data.DataLoader(validation_data,
                                          batch_size=batch_size,
                                          shuffle=True)
###### define constant########
input_channels = 3
hidden_size =128
max_epochs =100
lr = 3e-4
beta =-0.1

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



def se_loss(Y, X):
    loss1 = torch.sum((X - Y)**2,1)
    return loss1


def MSE_loss(Y, X):
    msk = torch.tensor(X > -1e-6).float()
    ret = ((X- Y) ** 2)
    ret = torch.sum(ret)
    return ret 


SIGMA=1
def Gaussian_CE_loss(Y, X, beta, sigma=SIGMA):  # 784 for mnist

    term2 =se_loss(Y, X)
    term3 = 1-torch.exp((-beta / (2 * (sigma**2))) * term2)
    loss1 = torch.mean(term3/beta)
    #recon_x
    #w_variance = torch.sum(torch.pow(recon_x[:,:,:,:-1]*msk[:,:,:,:-1] - recon_x[:,:,:,1:]*msk[:,:,:,1:], 2))
    #h_variance = torch.sum(torch.pow(recon_x[:,:,:-1,:]*msk[:,:,:-1,:] - recon_x[:,:,1:,:]*msk[:,:,1:,:], 2))
    #loss = 0.5 * (h_variance + w_variance)

    return loss1

# Reconstruction + KL divergence losses summed over all elements and batch
def beta_loss_function(recon_x, x, beta,z):
    msk = torch.tensor(x > 1e-6).float()

    if beta !=0:
        # If beta is nonzero, use the beta entropy
        BBCE = Gaussian_CE_loss(recon_x.view(-1, 1), x.view(-1, 1), beta)
    else:
        # if beta is zero use binary cross entropy
        BBCE = MSE_loss(recon_x.view(-1, 3*64*64), x.view(-1, 3*64*64))

    # compute KL divergence
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    z_term=5
    z_loss= torch.mean(torch.sum((z.view(-1, 512*8*8)**2),(1)))
    
    return BBCE+z_term*1/z_loss



pay=0
train_loss=0
valid_loss=0
valid_loss_list, train_loss_list= [], []
for epoch in range(max_epochs):
    
    train_loss=0
    valid_loss=0
    #opt_enc=exp_lr_scheduler(opt_enc, epoch, lr_decay=0.1, lr_decay_epoch=20)
    for data in train_loader:
        batch_size = data.size()[0]

        #print (data.size())
        datav = Variable(data).cuda()
        #datav[l2,:,row2:row2+5,:]=0
        #model_children = list(G.children())
        regularize_loss=G.sparse_loss()
        reg_weight=0.01
        rec_enc,z = G(datav)
        beta_err=beta_loss_function(rec_enc, datav,beta,z) 
        err_enc = beta_err+regularize_loss*reg_weight
        opt_enc.zero_grad()
        err_enc.backward()
        opt_enc.step()
        train_loss+=beta_err.item()
    train_loss /= len(train_loader.dataset)



    G.eval()
    with torch.no_grad():
        for data in Validation_loader:
            data = Variable(data).cuda()
            valid_rec,z_valid = G(data)
            beta_err=beta_loss_function(valid_rec, data,beta,z) 
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
    rec_imgs,z = G(fixed_batch)
    show_and_save('results/Input_epoch_%d.png' % epoch ,make_grid((fixed_batch.data[:,2:3,:,:]).cpu(),8))
    show_and_save('results/rec_epoch_%d.png' % epoch ,make_grid((rec_imgs.data[:,2:3,:,:]).cpu(),8))
    show_and_save('results/Error_epoch_%d.png' % epoch ,make_grid((fixed_batch.data[:,2:3,:,:]-rec_imgs.data[:,2:3,:,:]).cpu(),8))

    #localtime = time.asctime( time.localtime(time.time()) )
    #D_real_list_np=(D_real_list).to('cpu')
save_model(epoch, G)    
plt.plot(train_loss_list, label="train loss")
plt.plot(valid_loss_list, label="validation loss")
plt.legend()
plt.show()  