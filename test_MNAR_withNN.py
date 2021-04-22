import pandas as pd
import numpy as np
from fn import *
import torch as th


from torch import randn, matmul, rand
from torch.autograd import grad
from mvnorm import hyperrectangle_integral
from mvnorm.autograd.multivariate_normal_cdf import BivariateNormalCDF
from tqdm import tqdm
import torch.optim as optim

from torch.distributions import normal
from torch.nn import functional as F

nb_points = 1000

a0 = 3
a1 = .5
a2 = .7
b0 = 1
b1 = .5
b2 = .4

c0 = 4
c1 = .3
c2 = .2

rho = 0.5


Y, C, T, delta, xi,  X, XS, probaDelta, probaXi, f, g = test_gen_data_exponential_Heckman_Mnar( nb_points, a0, a1, a2, b0, b1, b2,c0, c1, c2, rho)


device = "cpu"

X = th.tensor(X).float().to(device).unsqueeze(-1)
XS = th.tensor(XS).float().to(device).unsqueeze(-1)

delta = th.tensor(delta).float().to(device).unsqueeze(-1)
xi = th.tensor(xi).float().to(device).unsqueeze(-1)


dataloader = th.utils.data.DataLoader(th.stack([X,XS,delta,xi],1), batch_size=200, shuffle=True)


rho = rand(1,requires_grad=True)


class Neural_network_regression(th.nn.Module):

    # Constructeur qui initialise le modèle
    def __init__(self,d,h1,h2):
        super(Neural_network_regression, self).__init__()

        self.layer1 = th.nn.Linear(d, h1)
        self.layer2 = th.nn.Linear(h1, h2)
        self.layer3 = th.nn.Linear(h2, 1)

        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.layer3.reset_parameters()

    # Implémentation de la passe forward du modèle
    def forward(self, x):
        phi1 = F.tanh(self.layer1(x))
        phi2 = F.tanh(self.layer2(phi1))
        return self.layer3(phi2)

f_nn = Neural_network_regression(1,200,100)
g_nn = Neural_network_regression(1,200,100)



eta1 = 0.001

eta2 = 0.01



optimizer1 = optim.Adam(list(f_nn.parameters()) + list(g_nn.parameters()), lr=eta1)
optimizer2 = optim.Adam([rho] , lr=eta2)

nb_epochs = 1000

pbar = tqdm(range(nb_epochs))

for i in pbar:

    cpt_batch = 0

    for data in dataloader:

        X = data[:,0]
        XS = data[:,1]
        delta = data[:,2]
        xi = data[:, 3]

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        gXS = g_nn(XS).squeeze(-1)


        fX = f_nn(X).squeeze(-1)

        upper0 = -gXS
        upper1 = th.stack([gXS,fX],1)
        upper2 = th.stack([gXS,-fX],1)

        # print("np.stack((g,f),1)")
        # print(np.stack((g,f),1)[0])
        #
        # print("upper1")
        # print(upper1[0])


        maxpts = 25000
        abseps = 0.001
        releps = 0

        bivariateNormalCDF = BivariateNormalCDF.apply

        m = normal.Normal(th.tensor([0.0]), th.tensor([1.0]))

        sum0 = -((1-xi)*th.log(m.cdf(upper0))).sum()
        sum1 = -(delta * xi * th.log(bivariateNormalCDF(upper1, rho, maxpts, abseps, releps))).sum()
        sum2 = -((1-delta) * xi * th.log(bivariateNormalCDF(upper2, -rho, maxpts, abseps, releps))).sum()

        loss = (sum0 + sum1  + sum2)/nb_points

        l2_lambda = 0.0
        l2_reg = th.tensor(0.)
        for param in f_nn.parameters():
            l2_reg += th.norm(param)
        for param in g_nn.parameters():
            l2_reg += th.norm(param)

        global_loss = loss +  l2_lambda * l2_reg


        global_loss.backward()


        optimizer1.step()
        optimizer2.step()

        cpt_batch += 1

        pbar.set_postfix(iter=i, idx_batch = cpt_batch, sum0 = sum0.item(), sum1 = sum1.item(), sum2 = sum2.item(), loss = loss.item(), global_loss = global_loss.item(), rho = rho.item())



print("real proba delta")
print(probaDelta)

print("estimated proba delta")
print(m.cdf(fX))


print("real proba xi")
print(probaDelta)

print("estimated proba xi")
print(m.cdf(fX))