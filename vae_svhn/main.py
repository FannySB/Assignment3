from __future__ import print_function
import utils, torch, os
import numpy as np
import argparse
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import random
import matplotlib.pyplot as plt


# from dataloader import get_data_loader
from dataloader import dataloader
import pdb


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
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


# splitdata = get_data_loader('binarized_mnist', args.batch_size)
# train_loader = splitdata[0]
# val_loader =  splitdata[1]
# test_loader =  splitdata[2]
dataset = 'svhn'
input_size = 32
z_dim=100
epoch = args.epochs
sample_num = 100
batch_size = args.batch_size
save_dir = 'save'
result_dir = 'results'
dataset = 'svhn' #args.dataset
log_dir = 'log'
gpu_mode = True
model_name = 'vae' #args.gan_type
lambda_ = 10
n_critic = 5 
sample_z = torch.randn((batch_size, z_dim)).to(device)

train_loader, val_loader, test_loader = dataloader(dataset, input_size, batch_size)

class VAE(nn.Module):
    def __init__(self, input_dim=100, output_dim=3, input_size=32):
        super(VAE, self).__init__()


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
        self.encoder_out1 = nn.Linear(1024, z_dim)
        self.encoder_out2 = nn.Linear(1024, z_dim)


        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def encode(self, x):
        # h1 = F.relu(self.fc1(x))
        # return self.fc21(h1), self.fc22(h1)
        x = x.view(-1, 3, input_size, input_size)
        x = self.encoder(x)
        x = x.view(-1, 1024)
        return self.encoder_out1(x), self.encoder_out2(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, input_size*input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # pdb.set_trace()
    # BCE = F.mse_loss(recon_x.view(-1, input_size*input_size*3), x.view(-1, input_size*input_size*3))
    # BCE = F.binary_cross_entropy(recon_x.view(-1, input_size*input_size*3), x.view(-1, input_size*input_size*3))
    BCE = F.binary_cross_entropy(recon_x.view(-1, input_size*input_size*3), x.view(-1, input_size*input_size*3), reduction='sum')

    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD
    # return -(BCE - KLD)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, z_latent = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # with torch.no_grad():
        #     visualize_recon(recon_batch, batch_idx)
        #     visualize_z_eps(z_latent, batch_idx)
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
            
            if batch_idx == 1020:
                print('printing...')
                with torch.no_grad():
                    visualize_recon(recon_batch, epoch)
                    # visualize_z_eps(z_latent, epoch)



    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def valid(epoch):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar, z_latent = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('====> Valid set loss: {:.4f}'.format(test_loss))
    return test_loss


def visualize_recon(recon, epoch):
    dir_recon = 'reconstruction'
    if not os.path.exists(result_dir + '/' + dataset + '/' + model_name + '/' + dir_recon):
        os.makedirs(result_dir + '/' + dataset + '/' + model_name + '/' + dir_recon)

    tot_num_samples = min(sample_num, batch_size)
    image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

    samples = recon.cpu().data.numpy().transpose(0, 2, 3, 1)

    samples = (samples + 1) / 2
    utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        result_dir + '/' + dataset + '/' + model_name + '/' + dir_recon + '/' + model_name + '_epoch%03d' % epoch + '_recon.png')
   
def visualize_z_eps(z_latent, epoch):
    dir_recon = 'reconstruction'
    if not os.path.exists(result_dir + '/' + dataset + '/' + model_name + '/' + dir_recon):
        os.makedirs(result_dir + '/' + dataset + '/' + model_name + '/' + dir_recon)

    tot_num_samples = min(sample_num, batch_size)
    image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
    epsilon = 100

    z_latent = z_latent.view(-1, z_dim)
    # rand_dim = random.randint(0,99)
    rand_dim = 5

    # pdb.set_trace()
    z_trans = torch.transpose(z_latent, 0, 1)
    z_before = z_trans[:][:rand_dim]
    z_eps = z_trans[:][rand_dim] + epsilon
    z_after = z_trans[:][rand_dim+1:]
    z_final = torch.cat((z_before, z_eps.view(1, -1), z_after), 0)
    z_samples = torch.transpose(z_final, 0, 1)

    samples = model.decode(z_samples)
    samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)

    samples = (samples + 1) / 2
    utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        result_dir + '/' + dataset + '/' + model_name + '/' + dir_recon + '/' + model_name + '_epoch%03d' % epoch + '_eps' + str(epsilon) + '.png')

# def visualize_z(z_latent, epoch):
#     dir_recon = 'reconstruction'
#     if not os.path.exists(result_dir + '/' + dataset + '/' + model_name + '/' + dir_recon):
#         os.makedirs(result_dir + '/' + dataset + '/' + model_name + '/' + dir_recon)

#     tot_num_samples = min(sample_num, batch_size)
#     image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))


#     z_latent = z_latent.view(-1, z_dim)
#     samples = model.decode(z_latent)
#     samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)

#     samples = (samples + 1) / 2
#     utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
#                         result_dir + '/' + dataset + '/' + model_name + '/' + dir_recon + '/' + model_name + '_epoch%03d' % epoch + '_z.png')
  

# def visualize_results(epoch, fix=True):
#     # self.G.eval()
#     dir_res = 'sample_z_fix_'+ str(fix)
#     if not os.path.exists(result_dir + '/' + dataset + '/' + model_name + '/' + dir_res):
#         os.makedirs(result_dir + '/' + dataset + '/' + model_name + '/' + dir_res)

#     tot_num_samples = min(sample_num, batch_size)
#     image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

#     epsilon = 0
#     if fix:
#         """ fixed noise """
#         samples = model.decode(sample_z) + epsilon
#     else:
#         """ random noise """
#         sample_z_ = torch.randn((batch_size, z_dim)) + epsilon
#         sample_z_ = sample_z_.to(device)

#         samples = model.decoder(sample_z_)
    
#     samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)

#     samples = (samples + 1) / 2
#     utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
#                         result_dir + '/' + dataset + '/' + model_name + '/' + dir_res + '/' + model_name + '_epoch%03d' % epoch + '.png')


def save():
    save_path = os.path.join(save_dir, dataset, model_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, model_name + '.pkl'))

if __name__ == "__main__":
    
    losses = []
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        loss = valid(epoch)
        losses.append(loss)
        save()
    
    plt.plot(losses)
    plt.savefig('losses.png')
    print('==============End Training============')
    