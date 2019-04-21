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


class MLPJSD(nn.Module):

    """ MLP class to use when using the Jensen Shannon Divergence.
    """

    def __init__(self, input_dim=2, hidden_dim=20):
        super(MLPJSD, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.input_dim = input_dim

    def forward(self, x):

        x = x.view(-1, self.input_dim)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x


class MLPWD(nn.Module):

    """ MLP class to use when using the Wasserstein Distance.
    """

    def __init__(self, input_dim=2, hidden_dim=50):
        super(MLPWD, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.input_dim = input_dim

    def forward(self, x):

        x = x.view(-1, self.input_dim)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

########### Question 1.1 ############


def loss_js(model, input_x, input_y):

    """Loss function to use when using the Shannon Jensen Divergence.
    """

    out_x = model(input_x)
    out_y = model(input_y)

    sum_x = torch.sum(torch.log(out_x))
    sum_y = torch.sum(torch.log(1-out_y))

    total_x = sum_x/ (2*batch_size)
    total_y = sum_y/ (2*batch_size)

    loss = - (total_x + total_y + torch.log(torch.tensor(2).float()))

    return loss

########### Question 1.2 ############

def loss_wd(model, input_x, input_y, alpha):


    """Loss function to use when using the Wasserstein Divergence.
    """
    out_x = model(input_x)
    out_y = model(input_y)

    obj = ( torch.sum(out_x) - torch.sum(out_y) )/ batch_size

    a = torch.rand( (input_x.size(0),1) )
    input_z = a * input_x + (1-a) * input_y
    input_z.requires_grad_(True)

    out_z = model(input_z)
    gradient_z = autograd.grad(out_z.sum(), input_z, create_graph=True)[0]

    norm_gradient = torch.norm(gradient_z, dim=1)

    penalty = ( torch.sum((norm_gradient - 1).pow(2)) )/ batch_size

    obj = obj - alpha * penalty

    loss = - obj
    return loss


########### Question 1.3 ############


def train(loss_function='js', epoch=10, batch_size=512, phi=0.9, alpha=10):

    """Training loop of discriminator.
    """

    if loss_function =='js':
        model = MLPJSD()

    elif loss_function =='wd':
        model = MLPWD()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    for epoch in range(epoch):

        optimizer.zero_grad()

        x = samplers.distribution1(0, batch_size)

        for input_x in x:
            input_x = Variable(torch.from_numpy(input_x)).float()
            break

        y = samplers.distribution1(phi, batch_size)

        for input_y in y:
            input_y = Variable(torch.from_numpy(input_y)).float()
            break

        if loss_function == 'js':

            loss = loss_js(model, input_x, input_y)

        elif loss_function == 'wd':

            loss = loss_wd(model, input_x, input_y, alpha)

        loss.backward()
        optimizer.step()

        loss_print = - loss

        if(epoch%50) == 0:
            print('epoch: {}, train loss: {:.6f}'.format(
                epoch, loss_print))

    return model, loss_print


## Obtaining the graph for question 1.3 with Jensen Shannon divergence


batch_size = 512
phi = -1.0
js = []
losses = []
phi_ = []

while not phi > 1.0:
    print('phi: 'phi)
    phi_.append(phi)
    model, loss = train(loss_function = 'js', batch_size=batch_size, epoch=1000, phi=phi)
    losses.append(loss.data[0])
    phi += 0.1


print('losses', losses)
plt.clf()
plt.plot(phi_, losses)
plt.title('JSD in terms of phi')
plt.xlabel('Phi values')
plt.ylabel('JSD')
plt.savefig('JSD.png')
print('==============End============')


## Obtaining the graph for question 1.3 with Wasserstein divergence

batch_size = 512
phi = -1.0
js = []
losses = []
phi_ = []

while not phi > 1.0:
    print(phi)
    phi_.append(phi)
    model, loss = train(loss_function = 'wd', batch_size=batch_size, epoch=1000, phi=phi, alpha=10)
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
