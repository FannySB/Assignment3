from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from VAE import VAE
from numpy.linalg import inv
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
def test(epoch):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for i, (data) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                # save_image(comparison.cpu(), 'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


# Function to evaluate log-likelihood of p(x) with variational autoencoders
def loglike(model, x, z):

    N = lambda x: np.exp(-x ** 2 / 2.) / ((2 * np.pi) ** 0.5)
    n = np.size(z, axis=2)
    K = np.size(z, axis=1)
    exp = np.exp((x-mu).T.dot(inv(np.diag(logvar.exp())))).dot((x-mu))
    denom = (((2 * np.pi)**(n))* np.linalg.det(logvar.exp()))**(0.5)
    gauss_z = exp / denom

    bce = F.binary_cross_entropy(z, np.repeat(x.view(-1, 784), K, axis=1), reduction='sum')

    g_z = np.log(bce) + np.log(N(z)) - np.log(gauss_z)

    logli = g_z.max() + np.log( (np.exp(g_z - g_z.max())).sum(axis=1) ) - np.log(K)

    return logli



if __name__ == "__main__":

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        # with torch.no_grad():
            # sample = torch.randn(100, 256).to(device)
            # sample = model.decode(sample).to(device)
            # save_image(sample.view(64, 1, 28, 28),
                    #    'results/sample_' + str(epoch) + '.png')
