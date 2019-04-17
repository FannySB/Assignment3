from __future__ import print_function
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import matplotlib.pyplot as plt
import samplers as samplers
import math
import pdb


class MLPNet(nn.Module):
    def __init__(self, input_dim=2):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 1)
        self.input_dim = input_dim

    def forward(self, x):

        x = x.view(-1, self.input_dim)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x


def estimatedJSD(loss_function='js', epoch=10, batch_size=512, phi=0.9, alpha=10):
    model = MLPNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    for epoch in range(epoch):

        optimizer.zero_grad()

        x = samplers.distribution1(0, batch_size)

        for input_x in x:
            input_x = Variable(torch.from_numpy(input_x)).float()
            out_x = model(input_x)
            break

        y = samplers.distribution1(phi, batch_size)

        for input_y in y:
            input_y = Variable(torch.from_numpy(input_y)).float()
            out_y = model(input_y)
            break

        if loss_function == 'js':

            sum_x = torch.sum(torch.log(out_x))
            sum_y = torch.sum(torch.log(1-out_y))

            total_x = sum_x/ (2*batch_size)
            total_y = sum_y/ (2*batch_size)

            loss = - (total_x + total_y + torch.log(torch.tensor(2).float()))

        elif loss_function == 'wd':

            objective = out_x.mean() - out_y.mean()

            a = torch.rand((input_x.size(0),1))
            input_z = a * input_x + (1-a) * input_y
            input_z.requires_grad_(True)

            out_z = model(input_z)
            gradient_z = autograd.grad(out_z.sum(), input_z, create_graph=True)[0]

            #g_norm = []
            #for grad in gradient_z:
            #    g_norm.append(grad)
            #    g_norm = torch.cat(g_norm, dim=1)

            norm_gradient = torch.norm(gradient_z, dim=1)

            penalty = (norm_gradient - 1).pow(2).mean()

            objective = objective - alpha * penalty

            loss = - objective

        loss.backward()
        optimizer.step()

        loss_print = - loss

        if(epoch%50) == 0:
            print('epoch: {}, train loss: {:.6f}'.format(
                epoch, loss_print))

    return model, loss_print



batch_size = 512
phi = -1.0
js = []
losses = []
phi_ = []

while not phi > 1.0:
    print(phi)
    phi_.append(phi)
    model, loss = estimatedJSD(loss_function = 'wd', batch_size=batch_size, epoch=500, phi=phi, alpha=0)
    losses.append(loss.data[0])
    phi += 0.1


print('losses', losses)
plt.clf()
plt.plot(phi_, losses)
plt.title('WD in terms of phi')
plt.xlabel('Phi values')
plt.ylabel('WD')
plt.savefig('WD.png')
print('==============End============')
