from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
import scipy.io as spio
from scipy.optimize import fmin, fminbound
from keras.datasets import fashion_mnist
import math
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials,rand
import hyperopt as hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials,rand
import hyperopt as hyperopt
from pyclustertend import hopkins
from math import isnan
from pyclustertend import hopkins,assess_tendency_by_mean_metric_score,assess_tendency_by_metric

#beta = 0.005  #0.00005
#batch_size = 133


seed = 10004
epochs = 150
batch_size = 120
log_interval = 10
#beta_val = 0.005  # 0.005 #0.00005,  0.03, 0.005
CODE_SIZE = 20
SIGMA = 0.5

np.random.seed(seed)


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size',
                    type=int,
                    default=128,
                    metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs',
                    type=int,
                    default=10,
                    metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='enables CUDA training')
parser.add_argument('--seed',
                    type=int,
                    default=1,
                    metavar='S',
                    help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def create_data(frac_anom):

 
    (X, X_lab), (_test_images, _test_lab) = fashion_mnist.load_data()
    X_lab = np.array(X_lab)

    # find bags
    ind = np.isin(X_lab, (0, 1, 2, 3, 4, 5, 6, 8))  #(1, 5, 7, 9)
    X_lab_outliers = X_lab[ind]
    X_outliers = X[ind]

    # find sneaker and ankle boots
    ind = np.isin(X_lab, (7, 9))  # (0, 2, 3, 4, 6))  #
    X_lab = X_lab[ind]
    X = X[ind]

    # X = ((X / 255.0) > 0.01).astype(np.float)
    X = X / 255.0
    #    X = X[:10000, ]
    #    X_lab = X_lab[:10000, ]

    # test_images = test_images / 255

    Nsamp = np.int(np.rint(len(X) * frac_anom)) + 1

    X_outliers = X_outliers / 255.0

    #N=np.ones((10000,28,28))

    #X=np.concatenate((X,N),axis=0)
    X[:Nsamp, :, :] = X_outliers[:Nsamp, :, :]
    X_lab[:Nsamp] = 10

    X = np.clip(X, 0, 1)
    X_train, X_test, X_lab_train, X_lab_test = train_test_split(
        X, X_lab, test_size=0.2, random_state=10003)

    X_train, X_valid, X_lab_train, X_lab_valid = train_test_split(
        X_train,X_lab_train, test_size=0.1, random_state=10003)


    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

    X_valid = X_valid.reshape((len(X_valid), np.prod(X_valid.shape[1:])))

    train_data = []
    for i in range(len(X_train)):
        train_data.append(
            [torch.from_numpy(X_train[i]).float(), X_lab_train[i]])

    test_data = []
    for i in range(len(X_test)):
        test_data.append([torch.from_numpy(X_test[i]).float(), X_lab_test[i]])

    valid_data = []
    for i in range(len(X_valid)):
        valid_data.append(
            [torch.from_numpy(X_valid[i]).float(), X_lab_valid[i]])

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=len(test_data),
                                              shuffle=True)

    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=len(valid_data),
                                               shuffle=True)

    return train_loader, test_loader, valid_loader


class RVAE(nn.Module):
    def __init__(self):
        super(RVAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, CODE_SIZE)
        self.fc22 = nn.Linear(400, CODE_SIZE)
        self.fc3 = nn.Linear(CODE_SIZE, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        return self.decode(mu), mu

    def weight_reset(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

    def sparse_loss(self):
        loss = 0
        for class_obj in self.modules():
            if isinstance(class_obj, nn.Linear) :
                    if class_obj.out_features >class_obj.in_features:
                        loss += torch.mean((class_obj.weight.data.clone()) ** 2)
                #for j in range(len(model_children[i])):
            #values = F.relu((model_children[i](values)))
                #loss += torch.mean((values)**2)
                #loss=0
        return loss

#        self.fc1.reset_parameters()
#        self.fc21.reset_parameters()
#        self.fc1.reset_parameters()

model = RVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def SE_loss(Y, X):
    ret = (X - Y)**2
    ret = torch.sum(ret)
    return ret

def MSE_loss(Y, X):
    msk = torch.tensor(X > -1e-6).float()
    ret = ((X- Y) ** 2)
    ret = torch.sum(ret,1)
    return ret


# Reconstruction + KL divergence losses summed over all elements and batch
def beta_loss_function(recon_x, x,beta,z):

    BBCE = torch.mean(MSE_loss(recon_x.view(-1, 28*28),x.view(-1, 28*28)))
    #z_term=20000
    z_loss= torch.mean(torch.sum((z)**2,(1)))
    return BBCE+(1/beta)*z_loss


def train(epoch, beta_val):
    model.train()
    train_loss = 0
    #    for batch_idx, data in enumerate(train_loader):
    for batch_idx, (data, data_lab) in enumerate(train_loader):
        #    for batch_idx, data in enumerate(train_loader):
        #data = (data.gt(0.5).type(torch.FloatTensor)).to(device)
        data = (data).to(device)

        optimizer.zero_grad()
        recon_batch, mu= model(data)
        loss = beta_loss_function(recon_batch, data, beta_val,mu)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()


'''        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
'''


def model_reset():
    model.weight_reset()


def test(frac_anom, beta_val):
    model.eval()
    test_loss_total = 0
    test_loss_anom = 0
    num_anom = 0
    with torch.no_grad():
        for i, (data, data_lab) in enumerate(test_loader):
            #        data = (data.gt(0.5).type(torch.FloatTensor)).to(device)
            data = (data).to(device)

            recon_batch, mu = model(data)
            #recon_batch = torch.tensor(recon_batch > 0.5).float()
            anom_lab = data_lab == 10
            num_anom += np.sum(anom_lab.numpy())  # count number of anomalies
            anom_lab = (anom_lab[:, None].float()).to(device)

            test_loss_anom += MSE_loss(recon_batch * anom_lab,
                                      data * anom_lab).item()
            test_loss_total += MSE_loss(recon_batch, data).item()

            if i == 0:
                n = min(data.size(0), 100)
                samp = [96, 97, 99, 90, 14, 35, 53, 57]
                comparison = torch.cat([
                    data.view(len(recon_batch), 1, 28, 28)[samp],
                    recon_batch.view(len(recon_batch), 1, 28, 28)[samp]
                ])

                save_image(comparison.cpu(),
                           'results/letters_mnist_recon_' + str(beta_val) +
                           '_' + str(frac_anom) + '.png',
                           nrow=n)

        np.savez('results/letters_mnist_' + str(beta_val) + '_' +
                 str(frac_anom) + '.npz',
                 recon=recon_batch.cpu(),
                 data=data.cpu(),
                 anom_lab=anom_lab.cpu())

    test_loss_normals = (test_loss_total - test_loss_anom) / (
        len(test_loader.dataset) - num_anom)
    test_loss_anom /= num_anom
    test_loss_total /= len(test_loader.dataset)

    print('====> Test set loss: {:.4f}'.format(test_loss_total))

    return test_loss_total, test_loss_anom, test_loss_normals

def hopkins_m(in_X):
     
    X = in_X[0].detach().cpu().numpy()
    #X=np.reshape(X,(X.shape[0],1))
    CL_num,H=assess_tendency_by_metric(X, n_cluster=3)
    
    return H[0]

def valid():
    losslist=[]
    model.eval()
    valid_loss_total = 0
    valid_loss_anom = 0
    num_anom = 0
    with torch.no_grad():
        for i, (data, data_lab) in enumerate(valid_loader):
            data = (data).to(device)

            recon_batch, mu = model(data)
            anom_lab = data_lab == 10
            num_anom += np.sum(anom_lab.numpy())
            anom_lab = (anom_lab[:, None].float()).to(device)

            valid_loss_anom += SE_loss(recon_batch * anom_lab,
                                       data * anom_lab).item()
            valid_loss_total += SE_loss(recon_batch, data).item()
            losslist.append(((data - recon_batch)**2))

    valid_loss_normals = (valid_loss_total - valid_loss_anom) / (
        len(valid_loader.dataset) - num_anom)
    valid_loss_anom /= num_anom
    valid_loss_total /= len(valid_loader.dataset)
    H = hopkins_m(losslist)

    return H


def train_valid(params):
    beta_val = params['x']

    model_reset()
    for epoch in erange:
        train(epoch, beta_val=beta_val)


#    test(0.1, beta_val)
    v_ratio = valid()

    return (-v_ratio)

if __name__ == "__main__":



    erange = range(1, epochs + 1)
    frac_anom = 0.1  # [0.01, 0.05, 0.1]  #
    #anrange = np.arange(0, 0.11, 0.005)  # fraction of anomalies  #

    train_loader, test_loader, valid_loader = create_data(frac_anom)

    
    fspace = {
    'x': hp.loguniform('x', -12, -10)
    }    
    bopt = fmin(fn=train_valid, space=fspace, algo=tpe.suggest, max_evals=50)
    print(bopt)
