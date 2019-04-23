import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

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


        self.encoder_out1 = nn.Linear(256, 100)
        self.encoder_out2 = nn.Linear(256, 100)
        self.decoder_lin = nn.Linear(100, 256)

        self.decoder1 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(256, 64, kernel_size=5, padding=4),
            nn.ELU(),
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=2),
            nn.ELU(),
        )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=2),
            nn.Sigmoid()
        )

    def encode(self, x):

        x = x.view(-1, 1, 28, 28)
        x = self.encoder(x)
        x = x.view(-1, 256)
        return self.encoder_out1(x), self.encoder_out2(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):

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