# Maths modules
import numpy as np
import math
from scipy.stats import expon




#############################################
# Kernel function for regression estimation #
#############################################
def quad_kern(x):
    return np.where(np.absolute(x)<=1,15/16*(1-x**2)**2,0)

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

    #Compute the list of weights
    W = np.zeros(n)
    W2 = np.zeros(n)
   
    if(mode_km):
        W = 1/(1+np.arange(n))
    else:
        W = quad_kern((x_eval-x)/h)
    
    W = W/np.sum(W)



    #Compute the list of the cumulative weights for the product
    cum_W = 1-np.append([0],np.cumsum(W[np.arange(0,n-1)]))
    
    W2[cum_W>0] = 1-W[cum_W>0]/cum_W[cum_W>0]
    W2[cum_W<=0] = np.zeros(n)[cum_W<=0]

    W2 = np.sign(W2)*np.abs(W2)**p
    # W2 = W2**p

    # Compute the conditional survival function
    for t_eval in t:
        s = 1-np.prod(W2[obs<=t_eval])
        csf.append(s)

    return csf





def Beran_estimator(p,t,obs, x=None, x_eval=None, h=0.1, mode_test=False):


    n = p.shape[0]

    W = np.zeros((n))


    for i in range(n):
        if(mode_test):
            W[i] = 1/n
        else:
            W[i] = quad_kern((x_eval - x[i])/h)

    W = W / np.sum(W)

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



