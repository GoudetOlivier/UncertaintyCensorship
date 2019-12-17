import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt
import math

#Hello

def biquadratic_kernel(x):
    return np.where(x<=1,15/16*(1-x**2)**2,0)

def RBF_kernel(x):
    return np.exp(-0.1*x**2)


def generate_data(nb_iter,size, a0, a1, a2, b0, b1, b2):

    np.random.seed(0)

    x = np.random.uniform(low=0.0, high=1.0, size=(nb_iter, size))

    y = np.random.exponential(1/(a0+a1*x+a2*x**2),(nb_iter,size))
    c = np.random.exponential(1/(b0+b1*x+b2*x**2),(nb_iter,size))

    t = np.minimum(y,c)

    delta = np.where(y<=c,1,0)

    #p = np.ones((nb_iter,size))*0.7
    p = 1/(1+t)

    u = np.random.uniform(low=0.0, high=1.0, size=(nb_iter,size))

    xi = np.where(u < p, 1, 0)

    index = np.argsort(t, axis=1)

    for i in range(nb_iter):
        y[i,:] = y[i,index[i,:]]
        t[i, :] = t[i, index[i, :]]
        c[i, :] = c[i, index[i, :]]
        delta[i, :] = delta[i, index[i, :]]
        xi[i, :] = xi[i, index[i, :]]
        p[i,:] = p[i, index[i, :]]
        x[i, :] = x[i, index[i, :]]



    return t, c, y, delta,  xi, p, x


def Kaplan_Meier(value,t,list_y,isNum=False):

    cdf = []
    n = t.shape[0]
    cumV = 1
    list_cumV = []

    for i in range(1,n+1):

        if(isNum):
            v = (1 - value[i-1] / (n - i +1))
        else:
            v = (1 - 1 / (n - i+1))**value[i-1]

        cumV = cumV*v
        list_cumV.append(1- cumV)

    cpt = 0
    for y in list_y:
        while(cpt < n and y > t[cpt] ):
            cpt+=1
        if(cpt == 0):
            cdf.append(0)
        else:
            cdf.append(list_cumV[cpt-1])
    return cdf



def Beran_estimator(value,t,list_y, x=None, x_eval=None,mode_test=False):

    n = t.shape[0]

    W = np.zeros((t.shape[0]))

    for i in range(t.shape[0]):
        if(mode_test):
            W[i] = 1/n
        else:
            # W[i] = biquadratic_kernel(x_eval-x[i])
            W[i] = RBF_kernel(x_eval - x[i])

    W = W / np.sum(W)

    cdf = []

    cumV = 1
    list_cumV = []

    sumW = 0

    for i in range(1,n+1):

        if(i>1):
            sumW += W[i-1]

        v = (1 - W[i-1] / (1-sumW))**value[i-1]

        if(math.isnan(v)):
            v = 0

        cumV = cumV*v
        list_cumV.append(1- cumV)

    cpt = 0
    for y in list_y:
        while(cpt < n and y > t[cpt] ):
            cpt+=1
        if(cpt == 0):
            cdf.append(0)
        else:
            cdf.append(list_cumV[cpt-1])
    return cdf




def our_estimator_test_beran_only_proba(t, xi, delta, list_y,h,x,x_eval):

    sigma = xi * delta

    pi = np.zeros((t.shape[0]))
    q = np.zeros((t.shape[0]))

    for i in range(t.shape[0]):
        d = h * ((t - t[i]) ** 2+(x - x[i]) ** 2)
        kernel = np.exp(-d)
        # pi[i] = np.sum(kernel * sigma) / np.sum(kernel)

        pi[i] = np.sum(kernel * xi) / np.sum(kernel)
        q[i] = np.sum(kernel * sigma) / np.sum(kernel)


    return np.array(Beran_estimator(q/pi,t,list_y, x, x_eval,False))



def our_estimator_test_beran_mix_delta_proba(t, xi, delta, list_y,h,x,x_eval):

    sigma = xi * delta

    pi = np.zeros((t.shape[0]))
    q = np.zeros((t.shape[0]))

    for i in range(t.shape[0]):
        d = h * ((t - t[i]) ** 2+(x - x[i]) ** 2)
        kernel = np.exp(-d)
        # pi[i] = np.sum(kernel * sigma) / np.sum(kernel)

        pi[i] = np.sum(kernel * xi) / np.sum(kernel)
        q[i] = np.sum(kernel * sigma) / np.sum(kernel)


    mix_delta_p = np.zeros((delta.shape[0]))

    cpt = 0
    for i in range(t.shape[0]):
        if (xi[i] == 0):
            mix_delta_p[i] = q[i]/pi[i]
            cpt += 1
        else:
            mix_delta_p[i] = delta[i]

    return np.array(Beran_estimator(mix_delta_p,t,list_y, x, x_eval,False))





# def beran_estimator_only_p(t, xi, delta, list_y,x,x_eval):
#
#     sigma = xi * delta
#
#     pi = np.zeros((t.shape[0]))
#     q = np.zeros((t.shape[0]))
#
#     for i in range(t.shape[0]):
#         d = h * (t - t[i]) ** 2
#         kernel = np.exp(-d)
#         pi[i] = np.sum(kernel * xi) / np.sum(kernel)
#         q[i] = np.sum(kernel * sigma) / np.sum(kernel)
#
#     return np.array(Beran(q/pi, t,list_y,h, x,x_eval,False))


# def beran_estimator_mix_delta_p(t, xi, delta, list_y,h,x,x_eval):
#
#     sigma = xi * delta
#
#     pi = np.zeros((t.shape[0]))
#     q = np.zeros((t.shape[0]))
#
#     for i in range(t.shape[0]):
#         d = h * (t - t[i]) ** 2
#         kernel = np.exp(-d)
#         pi[i] = np.sum(kernel * xi) / np.sum(kernel)
#         q[i] = np.sum(kernel * sigma) / np.sum(kernel)
#
#     mix_delta_p = np.zeros((delta.shape[0]))
#
#     cpt = 0
#     for i in range(t.shape[0]):
#         if (xi[i] == 0):
#             mix_delta_p[i] = q[i]/pi[i]
#             cpt += 1
#         else:
#             mix_delta_p[i] = delta[i]
#
#     return np.array(Beran(mix_delta_p, t,list_y,h, x,x_eval,False))


nb_iter = 200
size = 200

x_eval = 0.5

a0 = 1
a1 = 10
a2 = 10

b0 = 0.5
b1 = 5
b2 = 5


list_measures = list(np.arange(0,5,0.1))

t, c, y, delta, xi, p, x = generate_data(nb_iter,size, a0, a1, a2, b0, b1, b2)

# RBF kernel bandwith
lambda_ = 0.1

uncensored_cumulative_hazard_curve = []

for measure in list_measures:
    uncensored_cumulative_hazard_curve.append(expon(scale = 1/(a0 + a1 * x_eval + a2 * x_eval ** 2)).cdf(measure))

cdf_naive_KM_estimator = np.zeros((len(list_measures)))
cdf_beran_estimator_without_missing_values = np.zeros((len(list_measures)))

cdf_beran_estimator_with_missing_data_proba = np.zeros((len(list_measures)))

cdf_beran_estimator_mix_delta_proba = np.zeros((len(list_measures)))



for j in range(nb_iter):
    print("iter : " + str(j))

    cdf_naive_KM_estimator += np.array(Kaplan_Meier(delta[j,:],t[j,:],list_measures,False))
    cdf_beran_estimator_without_missing_values += np.array(Beran_estimator(delta[j,:],t[j,:], list_measures, x[j,:], x_eval, False))

    cdf_beran_estimator_with_missing_data_proba  += our_estimator_test_beran_only_proba(t[j,:], xi[j,:], delta[j,:], list_measures, lambda_, x[j,:], x_eval)

    cdf_beran_estimator_mix_delta_proba += our_estimator_test_beran_mix_delta_proba(t[j,:], xi[j,:], delta[j,:], list_measures, lambda_, x[j,:], x_eval)


cdf_naive_KM_estimator = cdf_naive_KM_estimator/nb_iter
cdf_beran_estimator_without_missing_values = cdf_beran_estimator_without_missing_values/nb_iter

cdf_beran_estimator_with_missing_data_proba  = cdf_beran_estimator_with_missing_data_proba/nb_iter

cdf_beran_estimator_mix_delta_proba = cdf_beran_estimator_mix_delta_proba/nb_iter



plt.plot(list_measures,uncensored_cumulative_hazard_curve, label='True cdf')
plt.plot(list_measures,cdf_naive_KM_estimator, label='cdf_naive_KM_wihtout_missing_values')
plt.plot(list_measures,cdf_beran_estimator_without_missing_values, label='cdf_beran_wihtout_missing_values')

plt.plot(list_measures,cdf_beran_estimator_with_missing_data_proba, label='cdf_beran_with_missing_values_only_proba_estimator')
plt.plot(list_measures,cdf_beran_estimator_mix_delta_proba, label='cdf_beran_with_missing_values_mix_delta_proba')

plt.legend()
plt.show()


# #### MISE evaluation ####

uncensored_cumulative_hazard_curve = []


ground_truth_cdf = np.zeros((nb_iter, size))

estimated_cdf_naive_KM_estimator = np.zeros((nb_iter, size))
esimated_cdf_beran_estimator_without_missing_values = np.zeros((nb_iter, size))

esimated_cdf_beran_estimator_with_missing_data_proba = np.zeros((nb_iter, size))
esimated_cdf_beran_estimator_mix_delta_proba = np.zeros((nb_iter, size))

for j in range(nb_iter):
    print("iter : " + str(j))


    list_measures = list(y[j,:])
    list_measures.sort()


    estimated_cdf_naive_KM_estimator[j,:] = np.array(Kaplan_Meier(delta[j,:],t[j,:],list_measures,False))
    esimated_cdf_beran_estimator_without_missing_values[j,:] = np.array(Beran_estimator(delta[j,:],t[j,:], list_measures, x[j,:], x_eval, False))

    esimated_cdf_beran_estimator_with_missing_data_proba[j,:] = our_estimator_test_beran_only_proba(t[j,:], xi[j,:], delta[j,:], list_measures, lambda_, x[j,:], x_eval)

    esimated_cdf_beran_estimator_mix_delta_proba[j,:] = our_estimator_test_beran_mix_delta_proba(t[j,:], xi[j,:], delta[j,:], list_measures, lambda_, x[j,:], x_eval)

    for idx, measure in enumerate(list_measures):
        ground_truth_cdf[j, idx] = expon(scale=1/(a0+a1*x_eval+a2*x_eval**2)).cdf(measure)



MISE_naive_KM_estimator = np.sum((estimated_cdf_naive_KM_estimator - ground_truth_cdf)**2)/(nb_iter*size)
MISE_beran_estimator_without_missing_values = np.sum((esimated_cdf_beran_estimator_without_missing_values - ground_truth_cdf)**2)/(nb_iter*size)

MISE_beran_estimator_with_missing_data_proba = np.sum((esimated_cdf_beran_estimator_with_missing_data_proba - ground_truth_cdf)**2)/(nb_iter*size)
MISE_beran_estimator_mix_delta_proba = np.sum((esimated_cdf_beran_estimator_mix_delta_proba - ground_truth_cdf)**2)/(nb_iter*size)

print("MISE_naive_KM_wihtout_missing_values")
print(MISE_naive_KM_estimator)

print("MISE_beran_wihtout_missing_values")
print(MISE_beran_estimator_without_missing_values)

print("MISE beran_with_missing_values_only_proba_estimator")
print(MISE_beran_estimator_with_missing_data_proba)

print("MISE beran_with_missing_values_mix_delta_proba")
print(MISE_beran_estimator_mix_delta_proba)
