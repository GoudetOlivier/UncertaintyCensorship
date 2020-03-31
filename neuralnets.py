#Neural network modules
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from utils.linear3d import Linear3D
from torch.nn import functional as F

#########################
# Neural Network models #
#########################



class Net(nn.Module):

    def __init__(self,layers_size):
        super(Net, self).__init__()
        self.layers_size = layers_size
        layers = []

        for i in range(len(layers_size) - 1):
            layers.append(nn.Linear(layers_size[i], layers_size[i+1]))

            if(i != len(layers_size) - 2):
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class Net3D(nn.Module):

    def __init__(self,nb_iter, layers_size):
        super(Net3D, self).__init__()
        self.layers_size = layers_size
        layers = []

        for i in range(len(layers_size) - 1):

            #layers.append(nn.Dropout(p=0.3))

            layers.append(Linear3D(nb_iter, layers_size[i], layers_size[i + 1]))
            #layers.append(ChannelBatchNorm1d(nb_iter, layers_size[i + 1]))


            if (i != len(layers_size) - 2):
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

#
# class L1Loss(nn.Module):
#
#
#     def __init__(self, size_average=None, reduce=None, reduction='mean'):
#         super(L1Loss, self).__init__(size_average, reduce, reduction)
#
#     def forward(self, input, target):
#
#         return th.max(1-input,0)
#         return F.l1_loss(input, target)
#

class NNetWrapper():

    def __init__(self, nb_iter, layers_size, device, isParallel_run, type_loss):

        self.device = device
        self.isParallel_run = isParallel_run

        if(isParallel_run):
            self.net = Net3D(nb_iter,layers_size)
        else:
            self.net = Net(layers_size)

        self.net.to(device)

        #self.optimizer = torch.optim.SGD(self.net.parameters(),lr=.001)
        self.optimizer = torch.optim.Adam(self.net.parameters())

        if(type_loss == "BCEloss"):
            self.criterion = nn.BCELoss()
        elif(type_loss == "MSEloss"):
            self.criterion = nn.MSELoss()
        elif (type_loss == "L1loss"):
            self.criterion = nn.L1Loss()
        # elif (type_loss == "hinge_loss"):
        #     self.criterion = nn.L1Loss()



        self.net.reset_parameters()


    def fit(self,nb_epoch, X, y):


        print('Neural network training : epoch =', nb_epoch)

        if (self.isParallel_run):

            inputs_batch = torch.transpose(torch.tensor(X, dtype=torch.float, device=self.device), 0, 1)
            target_batch = torch.transpose(torch.tensor(y, dtype=torch.float, device=self.device), 0, 1)

        else:

            inputs_batch = torch.tensor(X, dtype=torch.float, device=self.device)
            target_batch = torch.tensor(y, dtype=torch.float, device=self.device)
            self.net.reset_parameters()

        pbar = tqdm(range(nb_epoch))

        """Ajouter mini batchs ? """

        for epoch in pbar:

            self.optimizer.zero_grad()

            outputs_batch = self.net(inputs_batch)

            outputs_batch = outputs_batch.squeeze(-1)

            loss = self.criterion(outputs_batch,target_batch)


            loss.backward()
            self.optimizer.step()

            pbar.set_postfix(loss=loss.item())


    def predict(self,X):

        if (self.isParallel_run):

            inputs_batch = torch.transpose(torch.tensor(X, dtype=torch.float, device=self.device), 0, 1)
            outputs_batch = self.net(inputs_batch).squeeze(2)

            return  torch.transpose(outputs_batch, 0, 1).cpu().data.numpy()

        else:

            inputs_batch = torch.tensor(X, dtype=torch.float, device=self.device)
            outputs_batch = self.net(inputs_batch)

            return outputs_batch.cpu().data.numpy()[:, 0]

