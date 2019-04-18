from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

# from dataloader import get_data_loader
from dataloader import dataloader
import pdb


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
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


# splitdata = get_data_loader('binarized_mnist', args.batch_size)
# train_loader = splitdata[0]
# val_loader =  splitdata[1]
# test_loader =  splitdata[2]
dataset = 'svhn'
input_size = 28
z_dim=100
train_loader, val_loader, test_loader = dataloader(dataset, input_size, args.batch_size)

epoch = args.epoch
sample_num = 100
batch_size = args.batch_size
save_dir = args.save_dir
result_dir = args.result_dir
dataset = 'svhn' #args.dataset
log_dir = args.log_dir
gpu_mode = args.gpu_mode
model_name = 'vae' #args.gan_type
input_size = args.input_size
z_dim = 100
lambda_ = 10
n_critic = 5 


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # self.fc1 = nn.Linear(784, 400)
        # self.fc21 = nn.Linear(400, 20)
        # self.fc22 = nn.Linear(400, 20)
        # self.fc3 = nn.Linear(20, 400)
        # self.fc4 = nn.Linear(400, 784)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 256, kernel_size=5),
            nn.ELU(),
        )
            # nn.Linear(256, 2*100)

        
        self.encoder_out1 = nn.Linear(256, z_dim)
        self.encoder_out2 = nn.Linear(256, z_dim)
        self.decoder_lin = nn.Linear(z_dim, 256)

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
            nn.Conv2d(16, 3, kernel_size=3, padding=2),
            nn.Sigmoid() # ????
        )

    def encode(self, x):
        # h1 = F.relu(self.fc1(x))
        # return self.fc21(h1), self.fc22(h1)
        x = x.view(-1, 3, 28, 28)
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
    # pdb.set_trace()
    BCE = F.mse_loss(recon_x.view(-1, 28*28*3), x.view(-1, 28*28*3))

    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD
    # return -(BCE - KLD)


def train(epoch):
    model.train()
    train_loss = 0

    # pdb.set_trace()
    for batch_idx, (data, _) in enumerate(train_loader):
        # pdb.set_trace()
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

        with torch.no_grad():
            self.visualize_results((batch_idx+1))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def visualize_results(self, epoch, fix=True):
    # self.G.eval()

    if not os.path.exists(args.result_dir + '/' + dataset + '/' + model_name):
        os.makedirs(args.result_dir + '/' + dataset + '/' + model_name)

    tot_num_samples = min(self.sample_num, self.batch_size)
    image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

    epsilon = 0
    if fix:
        """ fixed noise """
        samples = self.G(self.sample_z_) + epsilon
    else:
        """ random noise """
        sample_z_ = torch.randn((self.batch_size, z_dim)) + epsilon
        if self.gpu_mode:
            sample_z_ = sample_z_.cuda()

        samples = self.G(sample_z_)

    if self.gpu_mode:
        samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
    else:
        samples = samples.data.numpy().transpose(0, 2, 3, 1)

    samples = (samples + 1) / 2
    utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)