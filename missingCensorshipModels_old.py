import torch.optim as optim
from mvnorm.autograd.multivariate_normal_cdf import BivariateNormalCDF
import torch
from fn import *
from tqdm import tqdm
from torch.distributions import normal

#
# class LinearExp(torch.nn.Module):
#
#     def __init__(self, d):
#
#         super(LinearExp, self).__init__()
#
#         self.weightsY = torch.nn.Parameter(torch.FloatTensor(d))
#         self.weightsC = torch.nn.Parameter(torch.FloatTensor(d))
#
#         # self.sigma = sigma
#
#
#     def forward(self, X, T):
#
#         mu1 = torch.sum(self.weightsY * X,1)
#         mu2 = torch.sum(self.weightsC * X,1)
#
#         return torch.erfinv(2 * mu2 / (mu1 + mu2) - 1) * math.sqrt(2)
#
#     def reset_parameters(self):
#
#         self.weightsY.data.uniform_(0, 1)
#         self.weightsC.data.uniform_(0, 1)

class WeibullMechanism(torch.nn.Module):

    def __init__(self, d):
        super(WeibullMechanism, self).__init__()

        self.layer1 = torch.nn.Linear(d, 1)
        self.layer2 = torch.nn.Linear(d, 1)

        self.m = normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    # def icdf(self, value):
    #     return self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)

    def forward(self, X, T):
        # x = torch.cat([X, T.unsqueeze(-1)], 1)


        k = self.layer1(X).squeeze_(-1)


        lambda_ = self.layer2(X).squeeze_(-1)

        print("torch.min(T)")
        print(torch.min(T))



        proba =  (k*torch.pow(T,(k-1)))/(k*torch.pow(T,(k-1))+lambda_)


        f = torch.erfinv(2 * proba - 1) * math.sqrt(2)


        # print("f.size()")
        # print(f.size())

        return f.unsqueeze_(-1)

    def reset_parameters(self):
        self.layer1.reset_parameters();
        self.layer2.reset_parameters();


class Linear(torch.nn.Module):

    def __init__(self,  d):

        super(Linear, self).__init__()


        self.layer = torch.nn.Linear(d,1)

    def forward(self, X, T):

        x = torch.cat([X,T.unsqueeze(-1)],1)

        return self.layer(x)

    def reset_parameters(self):

        self.layer.reset_parameters();




class Neural_network_regression(torch.nn.Module):

    # Constructeur qui initialise le modèle
    def __init__(self,layers_size):
        super(Neural_network_regression, self).__init__()

        layers = []

        for i in range(len(layers_size) - 1):

            layers.append(torch.nn.Linear(layers_size[i], layers_size[i + 1]))


            if (i != len(layers_size) - 2):
                # layers.append(torch.nn.BatchNorm1d(layers_size[i + 1]))
                layers.append(torch.nn.Sigmoid())

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, T):

        x = T.unsqueeze(-1)

        return self.layers(x)

    def forward(self, X, T):


        x = torch.cat([X,T.unsqueeze(-1)],1)

        return self.layers(x)


    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()





class HeckMan_MNAR():

    def __init__(self, f, g, device, noCovariateMode):

        self.f = f.to(device)
        self.g = g.to(device)

        self.f.reset_parameters()
        self.g.reset_parameters()

        self.rho = torch.rand(1,requires_grad=True, device = device)
        self.device = device
        self.m = normal.Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))

        self.noCovariateMode = noCovariateMode


    def fit(self,  X, XS, T, delta, xi, probaDelta, eta1, eta2,  nb_epochs, batch_size):

        print("Coucou")
        if(X.ndim == 1):
            X = torch.tensor(X).float().to(self.device).unsqueeze(-1)
        else:
            X = torch.tensor(X).float().to(self.device)

        if (XS.ndim == 1):
            XS = torch.tensor(XS).float().to(self.device).unsqueeze(-1)
        else:
            XS = torch.tensor(XS).float().to(self.device)


        T = torch.tensor(T).float().to(self.device).unsqueeze(-1)
        delta = torch.tensor(delta).float().to(self.device).unsqueeze(-1)
        xi = torch.tensor(xi).float().to(self.device).unsqueeze(-1)

        probaDelta = torch.tensor(probaDelta).float().to(self.device).unsqueeze(-1)


        traindata = torch.cat([ X, XS, T, delta, xi, probaDelta], axis=1)



        dataloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True)


        optimizer1 = optim.Adam(list(self.f.parameters()) + list(self.g.parameters()), lr=eta1)

        optimizer2 = optim.Adam([self.rho], lr=eta2)


        maxpts = 25000
        abseps = 0.001
        releps = 0

        bivariateNormalCDF = BivariateNormalCDF.apply



        pbar = tqdm(range(nb_epochs))


        for i in pbar:

            cpt_batch = 0

            for data in dataloader:

                X_batch = data[:, :X.shape[1]]


                XS_batch = data[:, X.shape[1]:(X.shape[1]+XS.shape[1])]

                T_batch = data[:, X.shape[1]+XS.shape[1]]
                delta_batch = data[:, X.shape[1]+XS.shape[1] + 1]
                xi_batch = data[:, X.shape[1]+XS.shape[1] + 2]

                probaDelta_batch = data[:, X.shape[1]+XS.shape[1] + 3]


                optimizer1.zero_grad()
                optimizer2.zero_grad()

                if(self.noCovariateMode):
                    gXS = self.g(None,T_batch).squeeze(-1)
                    fX =  self.f(None,T_batch).squeeze(-1)
                else:
                    gXS = self.g(XS_batch,T_batch).squeeze(-1)
                    fX =  self.f(X_batch,T_batch).squeeze(-1)

                # if (cpt_batch == 0):
                #     print("probaDelta")
                #     print(probaDelta_batch[:10])
                #
                #     print("m.cdf(fX)")
                #     print(self.m.cdf(fX)[:10])

                # if(i%10 == 0 and i > 0):
                #     print("print(self.m.cdf(fX))")
                #     print(self.m.cdf(fX)[:10])
                #     print("probaDelta_batch")
                #     print(probaDelta_batch[:10])

                diff_proba = torch.abs(self.m.cdf(fX) - probaDelta_batch).mean()

                upper0 = -gXS
                upper1 = torch.stack([gXS,fX],1)
                upper2 = torch.stack([gXS,-fX],1)

 
                
                sum0 = -((1 - xi_batch) * torch.log(self.m.cdf(upper0))).sum() / X_batch.shape[0]
                sum1 = -(delta_batch * xi_batch * torch.log(
                    bivariateNormalCDF(upper1, self.rho, maxpts, abseps, releps, self.device))).sum() / X_batch.shape[0]
                sum2 = -((1 - delta_batch) * xi_batch * torch.log(
                    bivariateNormalCDF(upper2, -self.rho, maxpts, abseps, releps, self.device))).sum() / X_batch.shape[0]

                loss = sum0 + sum1 + sum2

                loss.backward()

                # print("OK")
                # torch.nn.utils.clip_grad_norm_([self.rho] + list(self.f.parameters()) + list(self.g.parameters()), 0.0000001)
                # print("self.rho.grad")
                # print(self.rho.grad)

                optimizer1.step()
                optimizer2.step()

                if (self.rho.data > 1):
                    self.rho.data[0] = 0.99
                elif(self.rho.data < -1):
                    self.rho.data[0] = -0.99

                pbar.set_postfix(iter=i, idx_batch = cpt_batch, sum0 = sum0.item(), sum1 = sum1.item(), sum2 = sum2.item(), loss = loss.item(), rho = self.rho.item(), diff_proba = diff_proba.item())


                cpt_batch += 1

            if(torch.isnan(loss)):
                break

    def predict(self, X, T):

        if (X.ndim == 1):
            X = torch.tensor(X).float().to(self.device).unsqueeze(-1)
        else:
            X = torch.tensor(X).float().to(self.device)

        T = torch.tensor(T).float().to(self.device)


        out = self.m.cdf(self.f(X,T)).detach().cpu().numpy().squeeze(-1)


        return out




class HeckMan_MNAR_two_steps():

    def __init__(self, f, g, device, noCovariateMode):

        self.f = f.to(device)
        self.g = g.to(device)

        self.f.reset_parameters()
        self.g.reset_parameters()

        self.rho = torch.rand(1, requires_grad=True, device=device)
        self.rho.data = self.rho.data*2-1

        print("rho init : " + str(self.rho.data))
        self.device = device

        self.m = normal.Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))

        self.noCovariateMode = noCovariateMode

    def reinit(self):
        self.f.reset_parameters()
        self.rho = torch.rand(1, requires_grad=True, device=self.device)
        print("rho init : " + str(self.rho.data))

    def fit_xi(self, XS, T,  xi, probaXi, eta1, nb_epochs, batch_size):


        if (XS.ndim == 1):
            XS = torch.tensor(XS).float().to(self.device).unsqueeze(-1)
        else:
            XS = torch.tensor(XS).float().to(self.device)

        T = torch.tensor(T).float().to(self.device).unsqueeze(-1)

        xi = torch.tensor(xi).float().to(self.device).unsqueeze(-1)

        probaXi = torch.tensor(probaXi).float().to(self.device).unsqueeze(-1)

        traindata = torch.cat([XS, T, xi, probaXi], axis=1)

        dataloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True)

        optimizer1 = optim.Adam(list(self.g.parameters()), lr=eta1)


        pbar = tqdm(range(nb_epochs))

        for i in pbar:

            cpt_batch = 0

            for data in dataloader:

                XS_batch = data[:, :XS.shape[1]]
                T_batch = data[:, XS.shape[1]]
                xi_batch = data[:, XS.shape[1] + 1]
                probaXi_batch = data[:, XS.shape[1] + 2]

                optimizer1.zero_grad()

                if(self.noCovariateMode):
                    gXS = self.g(T_batch).squeeze(-1)
                else:
                    gXS = self.g(XS_batch, T_batch).squeeze(-1)

                diff_proba = torch.abs(self.m.cdf(gXS) - probaXi_batch).mean()

                loss = -(xi_batch * torch.log(self.m.cdf(gXS)) + (1 - xi_batch) * torch.log(
                    1 - self.m.cdf(gXS))).mean()


                loss.backward()

                optimizer1.step()


                pbar.set_postfix(iter=i, idx_batch=cpt_batch, loss=loss.item(),  diff_proba=diff_proba.item())

                cpt_batch += 1

            if (torch.isnan(loss)):
                break


    def fit_delta_rho(self, X, XS, T, delta, xi, probaDelta, eta1, eta2, nb_epochs, batch_size):

        if (X.ndim == 1):
            X = torch.tensor(X).float().to(self.device).unsqueeze(-1)
        else:
            X = torch.tensor(X).float().to(self.device)

        if (XS.ndim == 1):
            XS = torch.tensor(XS).float().to(self.device).unsqueeze(-1)
        else:
            XS = torch.tensor(XS).float().to(self.device)

        T = torch.tensor(T).float().to(self.device).unsqueeze(-1)
        delta = torch.tensor(delta).float().to(self.device).unsqueeze(-1)
        xi = torch.tensor(xi).float().to(self.device).unsqueeze(-1)

        probaDelta = torch.tensor(probaDelta).float().to(self.device).unsqueeze(-1)

        traindata = torch.cat([X, XS, T, delta, xi, probaDelta], axis=1)

        dataloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True)

        optimizer1 = optim.Adam(list(self.f.parameters()), lr=eta1)

        optimizer2 = optim.Adam([self.rho], lr=eta2)

        maxpts = 25000
        abseps = 0.001
        releps = 0

        bivariateNormalCDF = BivariateNormalCDF.apply

        pbar = tqdm(range(nb_epochs))

        for i in pbar:

            cpt_batch = 0

            for data in dataloader:

                X_batch = data[:, :X.shape[1]]

                XS_batch = data[:, X.shape[1]:(X.shape[1] + XS.shape[1])]

                T_batch = data[:, X.shape[1] + XS.shape[1]]
                delta_batch = data[:, X.shape[1] + XS.shape[1] + 1]
                xi_batch = data[:, X.shape[1] + XS.shape[1] + 2]

                probaDelta_batch = data[:, X.shape[1] + XS.shape[1] + 3]

                optimizer1.zero_grad()
                optimizer2.zero_grad()

                if (self.noCovariateMode):
                    gXS = self.g(T_batch).squeeze(-1)
                    fX = self.f(T_batch).squeeze(-1)
                else:
                    gXS = self.g(XS_batch, T_batch).squeeze(-1)
                    fX = self.f(X_batch, T_batch).squeeze(-1)


                diff_proba = torch.abs(self.m.cdf(fX) - probaDelta_batch).mean()

                upper1 = torch.stack([gXS, fX], 1)
                upper2 = torch.stack([gXS, -fX], 1)

                #sum0 = -((1 - xi_batch) * torch.log(self.m.cdf(upper0))).sum() / X_batch.shape[0]
                sum1 = -(delta_batch * xi_batch * torch.log(
                    bivariateNormalCDF(upper1, self.rho, maxpts, abseps, releps, self.device))).sum() / X_batch.shape[0]
                sum2 = -((1 - delta_batch) * xi_batch * torch.log(
                    bivariateNormalCDF(upper2, -self.rho, maxpts, abseps, releps, self.device))).sum() / X_batch.shape[
                           0]

                loss =  sum1 + sum2

                loss.backward()

                #torch.nn.utils.clip_grad_norm_(self.f.parameters(), 0.0000001)

                optimizer1.step()





                optimizer2.step()

                if (self.rho.data > 1):
                    self.rho.data[0] = 0.99
                elif (self.rho.data < -1):
                    self.rho.data[0] = -0.99

                pbar.set_postfix(iter=i, idx_batch=cpt_batch, sum1=sum1.item(), sum2=sum2.item(),
                                 loss=loss.item(), rho=self.rho.item(), diff_proba=diff_proba.item())

                cpt_batch += 1

            if (torch.isnan(self.rho.data)):
                break

    def predict(self, X, T):

        if (X.ndim == 1):
            X = torch.tensor(X).float().to(self.device).unsqueeze(-1)
        else:
            X = torch.tensor(X).float().to(self.device)

        T = torch.tensor(T).float().to(self.device)

        if (self.noCovariateMode):
            out = self.m.cdf(self.f(None, T)).detach().cpu().numpy().squeeze(-1)
        else:
            out = self.m.cdf(self.f(X, T)).detach().cpu().numpy().squeeze(-1)

        return out



class HeckMan_MAR():

    def __init__(self, f, device, noCovariateMode):

        self.f = f.to(device)


        self.f.reset_parameters()

        self.device = device

        self.m = normal.Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))

        self.noCovariateMode = noCovariateMode
        
        
    def fit(self,  X,  T, delta, probaDelta, eta, nb_epochs, batch_size):





        if(X.ndim == 1):
            X = torch.tensor(X).float().to(self.device).unsqueeze(-1)
        else:
            X = torch.tensor(X).float().to(self.device)


        T = torch.tensor(T).float().to(self.device).unsqueeze(-1)
        delta = torch.tensor(delta).float().to(self.device).unsqueeze(-1)

        probaDelta = torch.tensor(probaDelta).float().to(self.device).unsqueeze(-1)


        traindata = torch.cat([ X, T, delta,  probaDelta], axis=1)



        dataloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True)


        optimizer = optim.Adam(list(self.f.parameters()), lr=eta)



        pbar = tqdm(range(nb_epochs))


        for i in pbar:

            cpt_batch = 0

            for data in dataloader:

                X_batch = data[:, :X.shape[1]]
                T_batch = data[:, X.shape[1]]
                delta_batch = data[:, X.shape[1] + 1]
                probaDelta_batch = data[:, X.shape[1] + 2]


                optimizer.zero_grad()

                if(self.noCovariateMode):
                    fX = self.f(T_batch).squeeze(-1)
                else:
                    fX =  self.f(X_batch,T_batch).squeeze(-1)



                diff_proba = torch.abs(self.m.cdf(fX) - probaDelta_batch).mean()

                loss = -(delta_batch * torch.log(self.m.cdf(fX)) + (1-delta_batch) * torch.log(1-self.m.cdf(fX))).mean()

                loss.backward()


                optimizer.step()

                pbar.set_postfix(iter=i, idx_batch = cpt_batch, loss = loss.item(), diff_proba = diff_proba.item())

                cpt_batch += 1

            if (torch.isnan(loss)):
                break

    def predict(self, X, T):

        if (X.ndim == 1):
            X = torch.tensor(X).float().to(self.device).unsqueeze(-1)
        else:
            X = torch.tensor(X).float().to(self.device)

        T = torch.tensor(T).float().to(self.device)

        if(self.noCovariateMode):
            out = self.m.cdf(self.f(None,T)).detach().cpu().numpy().squeeze(-1)
        else:
            out = self.m.cdf(self.f(X, T)).detach().cpu().numpy().squeeze(-1)

        return out


