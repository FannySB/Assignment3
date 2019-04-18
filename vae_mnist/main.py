from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dataloader import get_data_loader
import pdb


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
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

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # self.fc1 = nn.Linear(784, 400)
        # self.fc21 = nn.Linear(400, 20)
        # self.fc22 = nn.Linear(400, 20)
        # self.fc3 = nn.Linear(20, 400)
        # self.fc4 = nn.Linear(400, 784)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 256, kernel_size=5),
            nn.ELU(),
        )
            # nn.Linear(256, 2*100)

        
        self.encoder_out1 = nn.Linear(256, 100)
        self.encoder_out2 = nn.Linear(256, 100)
        self.decoder_lin = nn.Linear(100, 256)

        self.decoder1 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(256, 64, kernel_size=5, padding=4),
            nn.ELU(),
        )
            # nn.inter(scale_factor=2),  # mode='bilinear'),
            # Interpolate(scale_factor=2, mode='bilinear'),
        self.decoder2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=2),
            nn.ELU(),
        )
            # nn.UpsamplingBilinear2d(scale_factor=2),  # mode='bilinear'),
        self.decoder3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=2),
            nn.Sigmoid() # ????
        )

    def encode(self, x):
        # h1 = F.relu(self.fc1(x))
        # return self.fc21(h1), self.fc22(h1)
        x = x.view(-1, 1, 28, 28)
        x = self.encoder(x)
        x = x.view(-1, 256)
        return self.encoder_out1(x), self.encoder_out2(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        # h3 = F.relu(self.fc3(z))
        # return torch.sigmoid(self.fc4(h3))
        
        # z = z.view(-1, 256)
        z = self.decoder_lin(z)
        z = z.view(-1, 256, 1, 1)
        z = self.decoder1(z)
        z = F.interpolate(z, scale_factor=2, mode='bilinear', align_corners = True)
        z = self.decoder2(z)
        z = F.interpolate(z, scale_factor=2, mode='bilinear', align_corners = True)
        return self.decoder3(z)


    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(F.softmax(recon_x), x.view(-1, 784), reduction='sum')
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD
    # return -(BCE - KLD)


def train(epoch):
    model.train()
    train_loss = 0

    # pdb.set_trace()
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

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        # with torch.no_grad():
            # sample = torch.randn(100, 256).to(device)
            # sample = model.decode(sample).to(device)
            # save_image(sample.view(64, 1, 28, 28),
                    #    'results/sample_' + str(epoch) + '.png')