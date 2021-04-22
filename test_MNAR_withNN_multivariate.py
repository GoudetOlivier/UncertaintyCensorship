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
        phi1 = F.relu(self.layer1(x))
        phi2 = F.relu(self.layer2(phi1))
        return self.layer3(phi2)



nb_points = 10000


rho = 0.5


#Y, C, T, delta, xi,  X, XS, probaDelta, probaXi, f, g = test_gen_data_exponential_Heckman_Mnar( nb_points, a0, a1, a2, b0, b1, b2,c0, c1, c2, rho)

Y, C, T, delta, xi,  X, XS, probaDelta = test_gen_data_multivariate_model_Heckman_Mnar(10, nb_points,rho)

device = "cpu"

Y = Y[0]
C = C[0]
T = T[0]
delta = delta[0]
xi = xi[0]
X = X[0]
XS = XS[0]
probaDelta = probaDelta[0]

print("Y.shape")
print(Y.shape)

print("X.shape")
print(X.shape)

print("probaDelta")
print(np.mean(probaDelta))


T = th.tensor(T).float().to(device).unsqueeze(-1)




X = th.tensor(X).float().to(device)
XS = th.tensor(XS).float().to(device)

delta = th.tensor(delta).float().to(device).unsqueeze(-1)
xi = th.tensor(xi).float().to(device).unsqueeze(-1)

# f = th.tensor(f).float().to(device).unsqueeze(-1)
# g = th.tensor(g).float().to(device).unsqueeze(-1)

probaDelta = th.tensor(probaDelta).float().to(device).unsqueeze(-1)


print("frac delta obs")
print(th.sum(xi)/nb_points)

print("th.sum(delta * xi)")
print(th.sum(delta * xi))

print("th.sum((1-delta) * xi)")
print(th.sum((1-delta) * xi))



traindata = th.cat([T,X,XS,delta,xi,probaDelta], axis = 1)

dataloader = th.utils.data.DataLoader(traindata, batch_size=1000, shuffle=True)


rho = rand(1,requires_grad=True)




f_nn = Neural_network_regression(6,200,100)
g_nn = Neural_network_regression(6,200,100)

maxpts = 25000
abseps = 0.001
releps = 0

bivariateNormalCDF = BivariateNormalCDF.apply
m = normal.Normal(th.tensor([0.0]), th.tensor([1.0]))

eta = 0.001
# eta2 = 0.01



optimizer = optim.Adam([rho] +  list(f_nn.parameters()) + list(g_nn.parameters()), lr=eta)

# optimizer2 = optim.Adam([rho] , lr=eta2)


nb_epochs = 10000

pbar = tqdm(range(nb_epochs))

for i in pbar:

    cpt_batch = 0

    cpt = 0

    for data in dataloader:

        T = data[:, 0]


        X_batch = data[:,1:6]
        XS_batch = data[:,6:11]
        delta_batch = data[:,11]
        xi_batch = data[:, 12]
        probaDelta = data[:, 13]

        # print(X_batch.shape)
        # print(XS_batch.shape)

        optimizer.zero_grad()
        # optimizer2.zero_grad()




        gXS = g_nn(th.cat([XS_batch,T.unsqueeze(-1)],1)).squeeze(-1)

        fX = f_nn(th.cat([X_batch,T.unsqueeze(-1)],1)).squeeze(-1)


        upper0 = -gXS
        upper1 = th.stack([gXS,fX],1)
        upper2 = th.stack([gXS,-fX],1)

        if(i%10 == 0  and cpt == 0):

            print("probaDelta")
            print(probaDelta[:10])

            print("m.cdf(fX)")
            print(m.cdf(fX)[:10])

        diff_proba = th.abs(m.cdf(fX) - probaDelta).mean()


        cpt += 1

        sum0 = -((1-xi_batch)*th.log(m.cdf(upper0))).sum()/X_batch.shape[0]
        sum1 = -(delta_batch * xi_batch * th.log(bivariateNormalCDF(upper1, rho, maxpts, abseps, releps))).sum()/X_batch.shape[0]
        sum2 = -((1-delta_batch) * xi_batch * th.log(bivariateNormalCDF(upper2, -rho, maxpts, abseps, releps))).sum()/X_batch.shape[0]

        loss = sum0 + sum1  + sum2

        # l2_lambda = 0.0
        # l2_reg = th.tensor(0.)
        # for param in f_nn.parameters():
        #     l2_reg += th.norm(param)
        # for param in g_nn.parameters():
        #     l2_reg += th.norm(param)
        #
        # global_loss = loss +  l2_lambda * l2_reg
        global_loss = loss


        global_loss.backward()


        optimizer.step()

        if(rho.data > 1 ):
            rho.data = 1
        elif(rho.data < -1):
            rho.data = -1


        cpt_batch += 1

        pbar.set_postfix(iter=i, idx_batch = cpt_batch, sum0 = sum0.item(), sum1 = sum1.item(), sum2 = sum2.item(), loss = loss.item(), global_loss = global_loss.item(), rho = rho.item(), diff_proba=diff_proba.item())



# print("real proba delta")
# print(probaDelta[:10])
#
# print("estimated proba delta")
# print(m.cdf(f_nn(X)).squeeze(-1)[:10])
#
#
# print("real proba xi")
# print(probaDelta[:10])
#
# print("estimated proba xi")
# print(m.cdf(g_nn(XS)).squeeze(-1)[:10])