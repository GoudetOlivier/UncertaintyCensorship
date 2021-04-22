# Maths modules
import numpy as np
import math
from scipy.stats import expon
import warnings
from scipy import special
import scipy.stats

from sklearn import datasets
from sklearn import preprocessing
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn


#############################################
# Kernel function for regression estimation #
#############################################
def quad_kern(x):
    return np.where(np.absolute(x)<=1,15/16*(1-x**2)**2,0)






########################################################################################
# Data generator with exponetial functions, returned data are sorted acccording to obs #
########################################################################################
def gen_data_exponential_Heckman_Mnar(nb_iter,size, a0, a1, a2, b0, b1, b2,c0,c1,c2,rho):

    np.random.seed(0)

    X = np.random.uniform(low=0.0, high=1.0, size=( size))

    mu1 = 1/(a0+a1*X+a2*X**2)
    mu2 = 1/(b0+b1*X+b2*X**2)


    Y = np.random.exponential(1/(a0+a1*X+a2*X**2),(nb_iter,size))
    C = np.random.exponential(1/(b0+b1*X+b2*X**2),(nb_iter,size))

    T = np.minimum(Y,C)

    delta = np.where(Y<=C,1,0)

    probaDelta = mu2/(mu1 + mu2)

    f = norm.ppf(probaDelta)

    g = 1/(c0+c1*X+c2*X**2)
    #g = c0+c1*X

    mean = np.array([0, 0])

    covarianceDelta0 = np.matrix([[1, -rho], [-rho, 1]])
    covarianceDelta1 = np.matrix([[1, rho], [rho, 1]])

    distDelta0 = mvn(mean=mean, cov=covarianceDelta0)
    distDelta1 = mvn(mean=mean, cov=covarianceDelta1)


    probaXi = np.where( delta == 1, distDelta1.cdf(np.stack((g,f),2))/probaDelta, distDelta0.cdf(np.stack((g,-f),2))/(1-probaDelta))

    print("probaXi")
    print(probaXi)


    u = np.random.uniform(low=0.0, high=1.0, size=(nb_iter,size))

    xi = np.where(u < probaXi, 1, 0)

    print("xi")
    print(xi)

    index = np.argsort(T, axis=1)

    for i in range(nb_iter):
        Y[i,:] = Y[i,index[i,:]]
        T[i, :] = T[i, index[i, :]]
        C[i, :] = C[i, index[i, :]]
        delta[i, :] = delta[i, index[i, :]]
        xi[i, :] = xi[i, index[i, :]]
        X[i, :] = X[i, index[i, :]]
        probaDelta[i, :] = probaDelta[i, index[i, :]]
        probaXi[i, :] = probaXi[i, index[i, :]]


    return Y, C, T, delta, xi,  X, probaDelta, probaXi


def test_gen_data_exponential_Heckman_Mnar(size, a0, b0, c0,rho):

    # np.random.seed(0)

    X = np.random.uniform(low=0.0, high=1.0, size=( size))

    XS = np.random.uniform(low=0.0, high=1.0, size=( size))


    mu1 = a0*X
    mu2 = b0*X
    print("mu1")
    print(mu1)

    print("mu2")
    print(mu2)

    Y = np.random.exponential(mu1,(size))
    C = np.random.exponential(mu2,(size))

    T = np.minimum(Y,C)

    delta = np.where(Y<=C,1,0)

    probaDelta = mu2/(mu1 + mu2)

    print("probaDelta")
    print(probaDelta)

    f = norm.ppf(probaDelta)


    print("f")
    print(f)

    # g = 1/(c0+c1*XS+c2*XS**2)
    g = c0 * XS

    print("g")
    print(g)


    mean = np.array([0, 0])

    covarianceDelta0 = np.matrix([[1, -rho], [-rho, 1]])
    covarianceDelta1 = np.matrix([[1, rho], [rho, 1]])


    distDelta0 = mvn(mean=mean, cov=covarianceDelta0)
    distDelta1 = mvn(mean=mean, cov=covarianceDelta1)




    probaXi_cond_delta = np.where( delta == 1, distDelta1.cdf(np.stack((g,f),1))/probaDelta, distDelta0.cdf(np.stack((g,-f),1))/(1-probaDelta))

    print("probaXi_cond_delta")
    print(probaXi_cond_delta)

    u = np.random.uniform(low=0.0, high=1.0, size=(size))

    xi = np.where(u < probaXi_cond_delta, 1, 0)

    probaXi = norm.cdf(g)

    print("probaXi")
    print(probaXi)
    index = np.argsort(T, axis=0)

    Y[:] = Y[index[:]]
    C[:] = C[index[:]]
    T[:] = T[index[:]]
    delta[:] = delta[index[:]]
    xi[:] = xi[index[:]]
    X[:] = X[index[:]]
    XS[:] = XS[index[:]]
    probaDelta[:] = probaDelta[index[:]]
    probaXi[:] = probaXi[index[:]]
    f[:] = f[index[:]]
    g[:] = g[index[:]]


    return Y, C, T, delta, xi,  X, XS, probaDelta, probaXi, f, g



def test_gen_data_exponential_Heckman_Mnar(nb_iter, size, a0, b0, c0,rho):

    # np.random.seed(0)

    X = np.random.uniform(low=0.0, high=1.0, size=(nb_iter,size))

    XS = np.random.uniform(low=0.0, high=1.0, size=(nb_iter,size))


    mu1 = a0*X
    mu2 = b0*X


    Y = np.random.exponential(mu1,(nb_iter,size))
    C = np.random.exponential(mu2,(nb_iter,size))

    T = np.minimum(Y,C)

    delta = np.where(Y<=C,1,0)

    probaDelta = mu2/(mu1 + mu2)

    f = norm.ppf(probaDelta)

    g = c0 * XS

    mean = np.array([0, 0])

    covarianceDelta0 = np.matrix([[1, -rho], [-rho, 1]])
    covarianceDelta1 = np.matrix([[1, rho], [rho, 1]])

    distDelta0 = mvn(mean=mean, cov=covarianceDelta0)
    distDelta1 = mvn(mean=mean, cov=covarianceDelta1)

    probaXi_cond_delta = np.where( delta == 1, distDelta1.cdf(np.stack((g,f),2))/probaDelta, distDelta0.cdf(np.stack((g,-f),2))/(1-probaDelta))

    u = np.random.uniform(low=0.0, high=1.0, size=(nb_iter,size))

    xi = np.where(u < probaXi_cond_delta, 1, 0)

    index = np.argsort(T, axis=1)

    for i in range(nb_iter):
        Y[i,:] = Y[i,index[i,:]]
        C[i,:] = C[i,index[i,:]]
        T[i,:] = T[i,index[i,:]]
        delta[i,:] = delta[i,index[i,:]]
        xi[i,:] = xi[i,index[i,:]]
        X[i,:] = X[i,index[i,:]]
        XS[i,:] = XS[i,index[i,:]]
        probaDelta[i,:] = probaDelta[i,index[i,:]]

    return Y, C, T, delta, xi,  X, XS, probaDelta



def test_gen_data_multivariate_model_Heckman_Mnar(nb_iter, size, rho):



    X = np.random.uniform(low=0,high=1, size=(nb_iter, size, 5))
    XS = np.random.uniform(low=0, high=1, size=(nb_iter, size, 5))
    sigma_Y = 0.3
    mu_Y = 0.5 + 1/5*(np.sin(X[:,:,0])+ np.cos(X[:,:,1]) + X[:,:,2]**2  -0.1*np.exp(X[:,:,3]) + X[:,:,4])
    Y = mu_Y +  sigma_Y* np.random.randn(nb_iter, size)
    lambda_C = 4 + X[:,:, 0] ** 3 + 1.5 * np.cos(X[:,:, 1]) + 3*X[:,:,  2] ** 2 + np.log(X[:,:,  3] + 5) + 5*X[:,:,  4]


    # X = np.random.uniform(low=0,high=1, size=( size, 2))
    # XS = np.random.uniform(low=0, high=1, size=(size, 2))
    # sigma_Y = 0.3
    # mu_Y = 2  + 0.5*X[:,0] + 2*X[:,1]**2
    #
    # Y = mu_Y +  sigma_Y* np.random.randn( size)
    # lambda_C = 2 +0.7* X[:, 0] + 0.5*X[:, 1] ** 2


    C = scipy.stats.expon(loc=1/lambda_C).rvs(size=( nb_iter, size))



    T = np.minimum(Y,C)

    delta = np.where(Y<=C,1,0)



    f_Y = scipy.stats.norm(loc=mu_Y, scale=sigma_Y).pdf(T)
    f_C = scipy.stats.expon(loc=1/lambda_C).pdf(T)

    S_Y = 1 - scipy.stats.norm.cdf(T, loc=mu_Y, scale=sigma_Y)
    S_C = 1 - scipy.stats.expon(loc=1/lambda_C).cdf(T)

    probaDelta =  (f_Y * S_C)/(f_Y * S_C + f_C * S_Y)

    print("probaDelta")
    print(probaDelta)

    f = norm.ppf(probaDelta)

    print("np.min(f)")
    print(np.min(f))

    print("np.max(f)")
    print(np.max(f))

    g  = 0 + 1 / 5 * (np.sin(XS[:,:,  0]) + np.cos(XS[:,:, 1]) + XS[:,:,  2] ** 3 + np.exp(XS[:,:, 3]) + 0.8*XS[:,:, 4])

    # g = -0.5 + XS[:, 0] + 0.8 * XS[:, 1]**2

    print("np.min(g)")
    print(np.min(g))

    print("np.max(g)")
    print(np.max(g))


    mean = np.array([0, 0])

    covarianceDelta0 = np.matrix([[1, -rho], [-rho, 1]])
    covarianceDelta1 = np.matrix([[1, rho], [rho, 1]])

    distDelta0 = mvn(mean=mean, cov=covarianceDelta0)
    distDelta1 = mvn(mean=mean, cov=covarianceDelta1)

    probaXi_cond_delta = np.where(delta == 1, distDelta1.cdf(np.stack((g, f), 2)) / probaDelta,
                       distDelta0.cdf(np.stack((g, -f), 2)) / (1 - probaDelta))


    print("probaXi_cond_delta")
    print(probaXi_cond_delta)

    u = np.random.uniform(low=0.0, high=1.0, size=(nb_iter, size))

    xi = np.where(u < probaXi_cond_delta, 1, 0)


    index = np.argsort(T, axis=1)


    for i in range(nb_iter):
        Y[i,:] = Y[i,index[i,:]]
        C[i,:] = C[i,index[i,:]]
        T[i,:] = T[i,index[i,:]]
        delta[i,:] = delta[i,index[i,:]]
        xi[i,:] = xi[i,index[i,:]]
        X[i,:] = X[i,index[i,:]]
        XS[i,:] = XS[i,index[i,:]]
        probaDelta[i,:] = probaDelta[i,index[i,:]]

    return Y, C, T, delta, xi,  X, XS, probaDelta




########################################################################################
# Data generator with exponetial functions, returned data are sorted acccording to obs #
########################################################################################
def gen_data_exponential(nb_iter,size, a0, a1, a2, b0, b1, b2):

    #np.random.seed(0)

    p = 0.5

    x = np.random.uniform(low=0.0, high=1.0, size=(nb_iter, size))

    mu1 = 1/(a0+a1*x+a2*x**2)
    mu2 = 1/(b0+b1*x+b2*x**2)


    surv = np.random.exponential(1/(a0+a1*x+a2*x**2),(nb_iter,size))
    censor = np.random.exponential(1/(b0+b1*x+b2*x**2),(nb_iter,size))

    obs = np.minimum(surv,censor)

    delta = np.where(surv<=censor,1,0)

    proba = mu2/(mu1 + mu2)


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
        proba[i, :] = proba[i, index[i, :]]


    return surv, censor, obs, delta,  xi, x, proba


##################################################################################################################
# Data generator according to Weibull distribution with covariate https://rdrr.io/cran/npcure/man/beran.html     #
##################################################################################################################
def gen_data_weibull(nb_iter,size):

    #np.random.seed(0)
    p = 0.5

    x = np.random.uniform(low=0,high=1, size=(nb_iter, size))

    surv = np.random.weibull(0.5*(x + 4))

    censor = np.random.exponential(1,(nb_iter,size))

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


    W2 = np.zeros(n)

    if(x.ndim==2):
        distance = np.linalg.norm(x_eval - x, axis=1)
    else:
        distance = x - x_eval

    if(mode_km):
        W = 1/(1+np.arange(n))
    else:
        W = quad_kern(distance/h)

    sum_W = np.sum(W)

    if(sum_W > 0):
        W = W/sum_W
    else:
        W = np.ones(n)/n

    #Compute the list of the cumulative weights for the product
    cum_W = 1-np.append([0],np.cumsum(W[np.arange(0,n-1)]))


    W2[cum_W>0] = 1-W[cum_W>0]/cum_W[cum_W>0]
    W2[cum_W<=0] = np.zeros(n)[cum_W<=0]

    W2 = np.sign(W2)*np.abs(W2)**p

    # Compute the conditional survival function
    for t_eval in t:

        s = 1-np.prod(W2[obs<=t_eval])
        csf.append(s)

    return csf


########################################################################
# Select bandwith of beran by crossvalidation (Geerdens et al. (2018)) #
########################################################################
def cross_val_beran(n,obs, delta, p, x,list_h,k):

    print("cross val " + str(k))

    ind_usefull_pair = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            if((delta[i] == 1 and delta[j] == 1)or (delta[i]==1 and obs[i] <= obs[j]) or (delta[j]==1 and obs[j] <= obs[i]) or (i==j)):
                ind_usefull_pair[i,j] = 1

    best_score = 99999999999999
    best_h = -1

    for h in list_h:

        score = 0

        for i in range(n):

            obs_del_i = np.delete(obs, i, 0)
            p_del_i = np.delete(p, i, 0)
            x_del_i = np.delete(x, i, 0)

            estimated_cdf_del_i = beran_estimator(obs, obs_del_i, p_del_i, x =  x_del_i ,x_eval =  x[i],  h = h)
            idx = np.where(obs[i] <= obs, 1, 0)
            score += np.sum(ind_usefull_pair[i,:]*(idx - estimated_cdf_del_i)**2)

        print("bandwith : " + str(h) + " score : " + str(score))

        if (score < best_score):

            best_h = h
            best_score = score

    return best_h



###################################################################################
# Select bandwith of beran by crossvalidation using proba instead of delta        #
###################################################################################
def cross_val_beran_proba(n,obs, p, x,list_h,k):

    print("cross val " + str(k))

    ind_usefull_pair = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            if(obs[i] <= obs[j]):
                ind_usefull_pair[i,j] = p[i]
            else:
                ind_usefull_pair[i, j] = p[j]

    best_score = 99999999999999
    best_h = -1

    for h in list_h:

        score = 0

        for i in range(n):

            obs_del_i = np.delete(obs, i, 0)
            p_del_i = np.delete(p, i, 0)
            x_del_i = np.delete(x, i, 0)

            estimated_cdf_del_i = beran_estimator(obs, obs_del_i, p_del_i, x =  x_del_i ,x_eval =  x[i],  h = h)
            idx = np.where(obs[i] <= obs, 1, 0)
            score += np.sum(ind_usefull_pair[i,:]*(idx - estimated_cdf_del_i)**2)

        print("bandwith : " + str(h) + " score : " + str(score))

        if (score < best_score):

            best_h = h
            best_score = score

    return best_h


###########################################################
# Data generator according to multivariate distribution.  #
###########################################################

def gen_data_multivariate_model(nb_iter,size):

    p = 0.5

    x = np.random.uniform(low=0,high=1, size=( nb_iter, size, 5))

    sigma_surv = 0.3

    mu_surv = 1 + 1/5*(np.sin(x[:,:,0])+ np.cos(x[:,:,1]) + x[:,:,2]**2 + np.exp(x[:,:,3]) + x[:,:,4])

    surv = mu_surv +  sigma_surv* np.random.randn(nb_iter, size)

    lambda_ = 3 + x[:, :, 0] ** 3 + 0.3 * np.cos(x[:, :, 1]) + x[:, :, 2] ** 2 + np.log(x[:, :, 3] + 0.1) + x[:, :, 4]

    censor = scipy.stats.expon(loc=1/lambda_).rvs(size=(nb_iter, size))

    obs = np.minimum(surv,censor)

    delta = np.where(surv<=censor,1,0)


    f_Y = scipy.stats.norm(loc=mu_surv, scale=sigma_surv).pdf(obs)
    f_C = scipy.stats.expon(loc=1/lambda_).pdf(obs)

    S_Y = 1 - scipy.stats.norm.cdf(obs, loc=mu_surv, scale=sigma_surv)
    S_C = 1 - scipy.stats.expon(loc=1/lambda_).cdf(obs)

    proba =  (f_Y * S_C)/(f_Y * S_C + f_C * S_Y)

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
        proba[i, :] = proba[i, index[i, :]]


    return surv, censor, obs, delta,  xi, x, proba


def gen_data_boston_housing(nb_iter):

    p = 0.5

    boston = datasets.load_boston()
    x, y = boston.data, boston.target


    x = preprocessing.scale(x)
    y =  preprocessing.scale(y)

    print(x.shape)
    print(y.shape)

    x = np.tile(x,(nb_iter,1,1))
    surv = np.tile(y,(nb_iter,1))

    print(x.shape)
    print(surv.shape)

    censor = scipy.stats.expon(loc=1/50).rvs(size=(nb_iter, surv.shape[1]))

    print("censor")
    print(censor)

    obs = np.minimum(surv,censor)

    delta = np.where(surv<=censor,1,0)

    print("delta")
    print(delta[0])

    print("mean delta")
    print(np.mean(delta))


    u = np.random.uniform(low=0.0, high=1.0, size=(nb_iter,surv.shape[1]))

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


def beran_estimator(t,obs,p,x=None, x_eval=None, h=0.1, mode_test=False):

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    n = p.shape[0]


    if (x.ndim == 2):
        distance = np.linalg.norm(x_eval-x, axis=1)
    else:
        distance = x - x_eval


    if (mode_test):
        W = 1 / (1 + np.arange(n))
    else:
        W = quad_kern(distance / h)

    sum_W = np.sum(W)

    if(sum_W > 0):
        W = W/sum_W
    else:
        W = np.ones(n)/n

    cdf = []

    cumV = 1
    list_cumV = []

    sumW = 0

    for i in range(n):


        v = (1 - W[i] / (1-sumW))**p[i]

        if(math.isnan(v)):
            v = 0

        sumW += W[i]

        cumV = cumV*v

        list_cumV.append(1- cumV)

    cpt = 0



    for t_eval in t:

        while(cpt < n and obs[cpt] < t_eval ):
            cpt+=1
        if(cpt == 0):
            cdf.append(0)
        else:
            cdf.append(list_cumV[cpt-1])

    return np.array(cdf)




def empirical_cdf(y, t_vect):

    cdf = np.zeros(t_vect.shape[0])

    index = np.argsort(y)
    y = y[index]

    for idx, t in enumerate(t_vect):

        cdf[idx] = np.sum(np.where(y<=t,1,0))/y.shape[0]

    return cdf
