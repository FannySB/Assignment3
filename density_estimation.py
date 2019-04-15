#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""


from __future__ import print_function
import numpy as np
import torch 
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import samplers as samplers
import math

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
# exact
xx = np.linspace(-5,5,1000)
N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.plot(xx, N(xx))



############### import the sampler ``samplers.distribution4'' 
############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######
batch_size = 512

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(batch_size, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 128)

    def forward(self, x):
        x = x.view(-1, batch_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x




def estimatedJSD(x, y):
    model = MLP()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    epoch = 10
    cpt = -1.0
    phi_array = []
    while not cpt > 1.0:
        cpt += 0.1
        phi_array.append(cpt)

    for epoch in range(epoch):
        for phi in phi_array:
            optimizer.zero_grad()
            phi = 1
            x = samplers.distribution1(0,batch_size)
            y = samplers.distribution1(phi, batch_size)

            out_x = model(x)
            out_y = model(y)

            for cpt in range(batch_size):
                total = x[cpt] + y[cpt]
                sum_y = math.log(1 - out_y[cpt])
                sum_x = math.log(out_x[cpt])

            total_x = sum_x/(2 * batch_size)
            total_y = sum_y/(2 * batch_size)
            loss = - (total_x + total_y + math.log(2, math.e))

def estimatedWasserstein():
    return 0



def D(x):
    #MLP 
    #call obj fct
    y= 0 #(0,1)
    return y

def JSD(x, y):


 
















############### plotting things
############### (1) plot the output of your trained discriminator 
############### (2) plot the estimated density contrasted with the true density



r = xx # evaluate xx using your discriminator; replace xx with the output
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(xx,r)
plt.title(r'$D(x)$')

estimate = np.ones_like(xx)*0.2 # estimate the density of distribution4 (on xx) using the discriminator; 
                                # replace "np.ones_like(xx)*0." with your estimate
plt.subplot(1,2,2)
plt.plot(xx,estimate)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.legend(['Estimated','True'])
plt.title('Estimated vs True')











