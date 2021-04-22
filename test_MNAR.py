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
import random

nb_points = 1000

a0 = 1
b0 = 1
c0 = 2

rho = 0.5

Y, C, T, delta, xi,  X, XS, probaDelta, probaXi, f, g = test_gen_data_exponential_Heckman_Mnar( nb_points, a0,  b0, c0, rho)

device = "cpu"

X = th.tensor(X).to(device)
XS = th.tensor(XS).to(device)
delta = th.tensor(delta).to(device)
xi = th.tensor(xi).to(device)

maxpts = 25000
abseps = 0.001
releps = 0

bivariateNormalCDF = BivariateNormalCDF.apply


print("frac delta obs")
print(th.sum(xi)/nb_points)

print("th.sum(delta * xi)")
print(th.sum(delta * xi))

print("th.sum((1-delta) * xi)")
print(th.sum((1-delta) * xi))


weight = np.random.uniform(0,1,4)

weight = th.tensor(weight).to(device)

w = th.nn.Parameter(weight, requires_grad=True)

eta = 0.01

optimizer = optim.Adam([w] , lr=eta)

nb_epochs = 0

pbar = tqdm(range(nb_epochs))


for i in pbar:

    optimizer.zero_grad()

    mu1 = w[0] * X
    mu2 = w[1] * X

    fX = th.erfinv(2 * mu2 / (mu1 + mu2) - 1) * math.sqrt(2)

    gXs = w[2]*XS

    upper0 = -gXs
    upper1 = th.stack([gXs,fX],1)
    upper2 = th.stack([gXs,-fX],1)

    rho = w[3]

    m = normal.Normal(th.tensor([0.0]), th.tensor([1.0]))

    sum0 = -((1-xi)*th.log(m.cdf(upper0))).sum()
    sum1 = -(delta * xi * th.log(bivariateNormalCDF(upper1, rho, maxpts, abseps, releps))).sum()
    sum2 = -((1-delta) * xi * th.log(bivariateNormalCDF(upper2, -rho, maxpts, abseps, releps))).sum()

    loss = sum0 + sum1  + sum2

    loss.backward()

    optimizer.step()

    pbar.set_postfix(iter=i, sum0 = sum0.item(), sum1 = sum1.item(), sum2 = sum2.item(), loss = loss.item(), w = w, rho = rho.item())



