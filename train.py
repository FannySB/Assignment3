from __future__ import print_function
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
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
        #x = torch.log(torch.tensor(2).float()) - torch.log(1 + torch.exp(-x))
        x = F.sigmoid(x)
        return x



def estimatedJSD(epoch=10, batch_size=512, phi=0.9):
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


        sum_x = torch.sum(torch.log(out_x))
        sum_y = torch.sum(torch.log(1-out_y))

        total_x = sum_x/ (2*batch_size)
        total_y = sum_y/ (2*batch_size)

        loss = - (total_x + total_y + torch.log(torch.tensor(2).float()))

        loss.backward()
        optimizer.step()

        if(epoch%50) == 0:
            print('epoch: {}, train loss: {:.6f}'.format(
                epoch, loss))

    return model, loss

def compute_js(D, batch_size):
    sum_x= torch.sum(torch.log(D))
    sum_y= torch.sum(torch.log(1-D))
    return sum_x/(2*batch_size) + sum_y/(2*batch_size) + torch.log(torch.tensor(2).float())


batch_size = 512
phi = -1.0
js = []
losses = []
phi_ = []
while not phi > 1.0:
    print(phi)
    phi_.append(phi)
    model, loss = estimatedJSD(batch_size=batch_size, epoch=1000, phi=phi)
    losses.append(-loss.data[0])

    x_new = samplers.distribution1(0, batch_size)
    for input_x_new in x_new:
        D_star = model(Variable(torch.from_numpy(input_x_new)).float())
        break

    js.append(compute_js(D_star, batch_size).data[0])
    phi += 0.1


print('js', js)
print('losses', losses)
plt.plot(phi_, js)
plt.savefig('js.png')
plt.clf()
plt.plot(phi_, losses)
plt.savefig('losses.png')
print('==============End============')
