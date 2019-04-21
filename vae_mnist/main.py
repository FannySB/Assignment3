from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from VAE import VAE
from numpy.linalg import inv
import numpy as np
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dataloader import get_data_loader
import pdb


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()





args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)


splitdata = get_data_loader('binarized_mnist', args.batch_size)
train_loader = splitdata[0]
val_loader =  splitdata[1]
test_loader =  splitdata[2]

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
def test(epoch, K):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for i, (data) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

            #Part to obtain the loglikelihood of x
            z = []
            M = data.size(dim=0)
            D = data.size(dim=2)*data.size(dim=3)
            L = mu.size(dim=1)
            mu = mu.cpu().numpy()
            logvar = logvar.cpu().numpy()

            for j in range(K):
                eps = np.random.normal(0, 1, (M, L))
                z.append(mu + np.exp(0.5*logvar)*eps)

            x = data.cpu().numpy().reshape(M,D)
            z = np.array(z).reshape(M, -1, L)
            loglikelihood = loglike_of_x(model, data, z)
            logli = loglikelihood.loglike()
            print('====> Loglikelihood of p(x): {:.4f}'.format,logli)
            #if i == 0:
                #n = min(data.size(0), 8)
                #comparison = torch.cat([data[:n],
                                      #recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                # save_image(comparison.cpu(), 'results/reconstruction_' + str(epoch) + '.png', nrow=n)



    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


# Function to evaluate log-likelihood of p(x) with variational autoencoders
class loglike_of_x():

    def __init__(self, model, x, z):
       self.model = model
       self.x = x
       self.z = z
       self.n = np.size(z, axis=2)
       self.K = np.size(z, axis=1)
       self.M = x.size(dim=0)
       self.D = x.size(dim=2)*x.size(dim=3)


    def _forward_loop(self):

        recon_x, mu, logvar = self.model(self.x)
        return recon_x, mu, logvar

    def _standard_normal(self, z):

        exp = np.exp( (-0.5)*( (z).T.dot(inv(np.diag(np.exp(np.ones(z.shape))))).dot((z))) )
        denom = (((2 * np.pi)**(self.n))* np.linalg.det(np.diag(np.exp(np.ones(z.shape)))))**(0.5)

        return exp/denom

    def _normal(self, z, mu, logvar):

        exp = np.exp( (-0.5)*( (z-mu).T.dot(inv(np.diag(np.exp(logvar)))).dot((z-mu))) )
        denom = (((2 * np.pi)**(self.n))* np.linalg.det(np.diag(np.exp(logvar))))**(0.5)
        return exp/denom

    def loglike(self):

        recon_x, self.mu, self.logvar = self._forward_loop()

        bce = []
        normal_standard = []
        normal = []

        # 3 values to calculate for this loop: bce, normal_standard and normal

        for i in range(self.K):
            for m in range(self.M):
                normal_standard.append(self._standard_normal(self.z[m,i,:]))
                normal.append(self._normal(self.z[m,i,:], self.mu[m].cpu().numpy(), self.logvar[m].cpu().numpy()))

            x_tilde = self.model.decode(torch.from_numpy(self.z[:,i,:]).float().to(device))
            bce.append(F.binary_cross_entropy(x_tilde, self.x, reduce=False, reduction='none').resize(self.M, self.D).sum(dim=1))


        #All have to be of dimension (M, K)
        bce = torch.stack(bce).cpu().numpy().reshape(self.M, self.K)
        normal_standard = np.array(normal_standard).reshape(self.M, self.K)
        normal = np.array(normal).reshape(self.M, self.K)

        #has to be of dimension (M, K)
        g_z = np.log(bce) + np.log(normal) - np.log(normal_standard)

        #has to be of dimension (M,)
        logli = g_z.max(axis=1) +\
                np.log( (np.exp(g_z - np.repeat(np.expand_dims(g_z.max(axis=1), axis=1), self.K, axis=1))).sum(axis=1) ) -\
                np.repeat(np.log(self.K), self.M, axis=0)

        return logli



if __name__ == "__main__":

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch, K=200)
        # with torch.no_grad():
            # sample = torch.randn(100, 256).to(device)
            # sample = model.decode(sample).to(device)
            # save_image(sample.view(64, 1, 28, 28),
                    #    'results/sample_' + str(epoch) + '.png')
