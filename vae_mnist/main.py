from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from VAE import VAE
from numpy.linalg import inv
from scipy.stats import multivariate_normal
import numpy as np
from torchvision import datasets, transforms
from torchvision.utils import save_image
from dataloader import get_data_loader



parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--load_model', type=str, default='',
                    help='path to model to load')
args = parser.parse_args()




#Initialization of environment
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


splitdata = get_data_loader('binarized_mnist', args.batch_size)
train_loader = splitdata[0]
val_loader = splitdata[1]
test_loader = splitdata[2]

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD
    # return -(BCE - KLD)


# Training function
def train(epoch):
    model.train()
    train_loss = 0

    for batch_idx, (data) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


# Testing function
def test(epoch, K, estimate_logli = 'False'):
    model.eval()
    test_loss = 0

    if estimate_logli == 'False':
        with torch.no_grad():
            for i, (data) in enumerate(test_loader):
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                test_loss += loss_function(recon_batch, data, mu, logvar).item()
    else:
        with torch.no_grad():
            logli = []
            mean_logli = []
            for i, (data) in enumerate(test_loader):
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                test_loss += loss_function(recon_batch, data, mu, logvar).item()

                #Part to obtain the loglikelihood of x
                z = []

                M = data.size(dim=0)
                D = data.size(dim=2)*data.size(dim=3)
                L = mu.size(dim=1)

                mu = mu.detach().cpu().numpy()
                logvar = logvar.detach().cpu().numpy()

                for j in range(K):
                    eps = np.random.normal(0, 1, (M, L))
                    z.append(mu + np.exp(0.5*logvar)*eps)

                x = data.cpu().numpy().reshape(M,D)
                z = np.stack(z)
                z = np.swapaxes(z, 0, 1)

                loglikelihood = loglike_of_x(model, data, z)
                logli.append(loglikelihood.loglike())
                mean_logli.append(logli[i].mean())



                print('minibatch :', i)
                print('====> Loglikelihood of p(x): mean of minibatch: ', logli[i].mean())


            print('====> Loglikelihood of p(x): mean of entire set: ', np.array(mean_logli).mean())



    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def valid(epoch, K, estimate_logli = 'False'):
    model.eval()
    test_loss = 0

    if estimate_logli == 'False':
        with torch.no_grad():
            for i, (data) in enumerate(val_loader):
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                test_loss += loss_function(recon_batch, data, mu, logvar).item()
    else:
        with torch.no_grad():
            logli = []
            mean_logli = []
            for i, (data) in enumerate(val_loader):
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                test_loss += loss_function(recon_batch, data, mu, logvar).item()

                #Part to obtain the loglikelihood of x
                z = []

                M = data.size(dim=0)
                D = data.size(dim=2)*data.size(dim=3)
                L = mu.size(dim=1)

                mu = mu.detach().cpu().numpy()
                logvar = logvar.detach().cpu().numpy()

                for j in range(K):
                    eps = np.random.normal(0, 1, (M, L))
                    z.append(mu + np.exp(0.5*logvar)*eps)

                x = data.cpu().numpy().reshape(M,D)
                z = np.stack(z)
                z = np.swapaxes(z, 0, 1)
                loglikelihood = loglike_of_x(model, data, z)
                logli.append(loglikelihood.loglike())
                mean_logli.append(logli[i].mean())

                print('minibatch :', i)
                print('====> Loglikelihood of p(x): mean of minibatch: ', logli[i].mean())


            print('====> Loglikelihood of p(x): {:.4f}'.format, np.array(mean_logli).mean())



    test_loss /= len(val_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))



# class to evaluate log-likelihood of p(x) with variational autoencoders
class loglike_of_x():

    def __init__(self, model, x, z):
       self.model = model
       self.x = x
       self.z = z
       self.n = np.size(z, axis=2)
       self.K = np.size(z, axis=1)
       self.M = x.size(dim=0)
       self.D = x.size(dim=2)*x.size(dim=3)
       self.L = z.shape[2]


    def _forward_loop(self):

        recon_x, mu, logvar = self.model(self.x)
        return recon_x, mu, logvar

    def _log_standard_normal(self, z, L):

        pi_term = -(L/2) * np.log(2*np.pi)

        return (-z**(2)*(0.5)).sum() + pi_term


    def _log_normal(self, z, mu, logvar, L):

        pi_term = -(L/2) * np.log(2*np.pi)
        stability = np.repeat(10**(-7), L, axis=0)

        return (-(z-mu)**2/(2*np.exp(logvar) + stability) - np.log(np.exp(0.5*logvar) + stability )).sum() + pi_term

    def loglike(self):

        recon_x, self.mu, self.logvar = self._forward_loop()

        bce = []
        log_normal_standard = []
        log_normal = []

        for i in range(self.K):
            for m in range(self.M):
                log_normal_standard.append(self._log_standard_normal(self.z[m,i,:], self.L))
                log_normal.append(self._log_normal(self.z[m,i,:], self.mu[m].detach().cpu().numpy(),\
                                                   self.logvar[m].detach().cpu().numpy(), self.L))

            x_tilde = self.model.decode(torch.from_numpy(self.z[:,i,:]).float().to(device))
            bce.append(-F.binary_cross_entropy(x_tilde, self.x, reduce=False, reduction='none').resize(self.M, self.D).sum(dim=1))

        #All have to be of dimension (M, K)
        bce = torch.stack(bce).detach().cpu().numpy()
        bce = np.swapaxes(bce, 0, 1)

        log_normal_standard = np.array(log_normal_standard).reshape(self.K, self.M)
        log_normal_standard = np.swapaxes(log_normal_standard, 0, 1)

        log_normal = np.array(log_normal).reshape(self.K, self.M)
        log_normal = np.swapaxes(log_normal, 0, 1)

        #has to be of dimension (M, K)
        w_z = bce + log_normal_standard - log_normal


        #has to be of dimension (M,)
        logli = w_z.max(axis=1) +\
                np.log( (np.exp(w_z - np.repeat(np.expand_dims(w_z.max(axis=1), axis=1), self.K, axis=1))).sum(axis=1) ) -\
                np.repeat(np.log(self.K), self.M, axis=0)


        return logli



if __name__ == "__main__":

    # First step to train the model for default number of epoch values: 20

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        valid(epoch, K=200, estimate_logli='False')
        test(epoch, K=200, estimate_logli='False')

    # Second step to save the model parameters and load them:
    # If saving model on cpu, use map_location='cpu'

    torch.save(model.state_dict(), args.load_model)
    model.load_state_dict(torch.load(args.load_model))
    #model.load_state_dict(torch.load(args.load_model, map_location='cpu'))

    # Third step to perform importance sampling

    valid(1, K=200, estimate_logli='True')
    test(1, K=200, estimate_logli='True')

