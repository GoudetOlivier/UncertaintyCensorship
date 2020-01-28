# Maths modules
import numpy as np
import math
from scipy.stats import expon

#Neural network modules
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#############################################
# Kernel function for regression estimation #
#############################################
def quad_kern(x):
    return np.where(np.absolute(x)<=1,15/16*(1-x**2)**2,0)

##############################################################
# Data generator, returned data are sorted acccording to obs #
##############################################################
def gen_data(nb_iter,size, a0, a1, a2, b0, b1, b2):

    #np.random.seed(0)

    p = 0.5

    x = np.random.uniform(low=0.0, high=1.0, size=(nb_iter, size))

    surv = np.random.exponential(1/(a0+a1*x+a2*x**2),(nb_iter,size))
    censor = np.random.exponential(1/(b0+b1*x+b2*x**2),(nb_iter,size))

    obs = np.minimum(surv,censor)

    delta = np.where(surv<=censor,1,0)

    u = np.random.uniform(low=0.0, high=1.0, size=(nb_iter,size))

    xi = np.where(u < p, 1, 0)

    index = np.argsort(obs, axis=1)

    for i in range(nb_iter):
        surv[i,:] = surv[i,index[i,:]]
        obs[i, :] = obs[i, index[i, :]]
        censor[i, :] = censor[i, index[i, :]]
        delta[i, :] = delta[i, index[i, :]]
        xi[i, :] = xi[i, index[i, :]]
        x[i, :] = x[i, index[i, :]]

    return surv, censor, obs, delta,  xi, x

#####################################################################
# Generalized Beran estimator for the conditional survival function #
#####################################################################
def gene_Beran(t, obs, p, x=None, x_eval=None, h=1, mode_km=False):

    n = len(obs)
    csf = []

    #Compute the list of weights
    W = np.zeros(n)
   
    if(mode_km):
        W = 1/(1+np.arange(n))
    else:
        W = quad_kern((x_eval-x)/h)
    
    W = W/np.sum(W)

    #Compute the list of the cumulative weights for the product
    cum_W = (1-W/np.append([1],1-np.cumsum(W[np.arange(1,n)])))**p

    # Compute the conditional survival function
    for i in range(len(t)):
    	csf.append(np.prod(cum_W[obs<=t[i]]))

    return csf

########################
# Neural Network model #
########################
# class Net(nn.Module):
#
#     def __init__(self,layers_size):
#         super(Net, self).__init__()
#         self.layers_size = layers_size
#         self.hidden = nn.ModuleList([nn.Linear(layers_size[i], layers_size[i+1]) for i in range(len(layers_size)-1)])
#
#     def forward(self, x):
#         if x.dim() == 0 :
#             x = torch.tensor(x.item()*np.ones(self.layers_size[0]),device=x.device)
#         for hidden_layer in self.hidden:
#             x = F.relu(hidden_layer(x))
#         return x


class Net(nn.Module):

    def __init__(self,layers_size):
        super(Net, self).__init__()
        self.layers_size = layers_size
        layers = []


        for i in range(len(layers_size) - 1):
            layers.append(nn.Linear(layers_size[i], layers_size[i+1]))
            if(i != len(layers_size) - 2):
                layers.append(nn.LeakyReLU(.2))


        self.layers = nn.Sequential(*layers)



    def forward(self, x):

        # if x.dim() == 0 :
        #     x = torch.tensor(x.item()*np.ones(self.layers_size[0]),device=x.device)
        # for hidden_layer in self.hidden:
        #     x = F.relu(hidden_layer(x))
        return self.layers(x)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()