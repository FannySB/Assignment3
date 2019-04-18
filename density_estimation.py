#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""


from __future__ import print_function
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
import samplers as samplers
import math
import pdb

use_cuda = torch.cuda.is_available()

# plot p0 and p1
plt.figure()

# empirical
xx = torch.randn(10000)
f = lambda x: torch.tanh(x*2+1) + x*0.75
d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
plt.hist(f(xx), 100, alpha=0.5, density=1)
plt.hist(xx, 100, alpha=0.5, density=1)
plt.xlim(-5,5)
plt.savefig('empirical.png')
# exact
plt.clf()
xx = np.linspace(-5,5,1000)
N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.plot(xx, N(xx))
plt.savefig('exact.png')



############### import the sampler ``samplers.distribution4''
############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######

class MLP(nn.Module):

    """ MLP class to use for Q.4.
    """

    def __init__(self, input_dim=2, hidden_dim=20):
        super(MLP, self).__init__()
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

def loss_function(model, input_x, input_y):

    """Loss function to use for Q1.4.
    """

    out_x = model(input_x)
    out_y = model(input_y)

    sum_x = torch.sum(torch.log(out_x))
    sum_y = torch.sum(torch.log(1-out_y))

    total_x = sum_x/ (batch_size)
    total_y = sum_y/ (batch_size)

    loss = - (total_x + total_y)

    return loss

def estimatef1(xx, model):

    """Computes the estimate of the f1 density using
    the procedure of Question 5 in the theory part.
    """
    input_xx = Variable(torch.from_numpy(xx)).float()

    fo = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
    f1 = (fo(xx).reshape(-1, 1)) * ( model(input_xx) * (1- model(input_xx)).pow(-1) ).detach().numpy()

    return f1



def train(epoch=10, batch_size=512, lr=1e-3):

    """Training loop of discriminator.
    """

    model = MLP(input_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=lr)


    for epoch in range(epoch):

        optimizer.zero_grad()

        x = samplers.distribution4(batch_size=512)
        for input_x in x:
            input_x = Variable(torch.from_numpy(input_x)).float()
            break

        y = samplers.distribution3(batch_size=512)
        for input_y in y:
            input_y = Variable(torch.from_numpy(input_y)).float()
            break

        loss = loss_function(model, input_x, input_y)

        loss.backward()
        optimizer.step()

        loss_print = - loss

        if(epoch%50) == 0:
            print('epoch: {}, train loss: {:.6f}'.format(
                epoch, loss_print))

    return model, loss_print


############### plotting things
############### (1) plot the output of your trained discriminator
############### (2) plot the estimated density contrasted with the true density
batch_size = 512
model, loss = train(batch_size=batch_size, epoch=1000)

out_xx = model(Variable(torch.from_numpy(xx)).float()).detach().numpy()# evaluate xx using your discriminator; replace xx with the output

plt.clf()
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(xx, out_xx)
plt.title(r'$D(x)$')

estimate = estimatef1(xx, model) # estimate the density of distribution4 (on xx) using the discriminator;
print(estimate.shape)                                # replace "np.ones_like(xx)*0." with your estimate
plt.subplot(1,2,2)
plt.plot(xx,estimate)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.legend(['Estimated','True'])
plt.title('Estimated vs True')
plt.savefig('Q1.4.png')
