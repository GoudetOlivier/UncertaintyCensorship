import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt

def generate_data(nb_iter,size):

    np.random.seed(0)

    y = np.random.exponential(1,(nb_iter,size))
    c = 5*np.random.uniform(low=0.0, high=1.0, size=(nb_iter,size))
    # c = np.random.exponential(2,(nb_iter,size))

    t = np.minimum(y,c)

    delta = np.where(y<=c,1,0)

    p = np.ones((nb_iter,size))*0.7

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

    return t, c, y, delta,  xi, p



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



def Subramanian_2004(t, xi, delta, list_y,h):

    sigma = xi*delta

    pi = np.zeros((t.shape[0]))
    q = np.zeros((t.shape[0]))

    for i in range(t.shape[0]):

        d = h*(t-t[i])**2
        kernel = np.exp(-d)
        pi[i] = np.sum(kernel*xi)/np.sum(kernel)
        q[i] = np.sum(kernel*sigma)/np.sum(kernel)

    return np.array(Kaplan_Meier(q/pi,t,list_y,False))



def Subramanian_2006(t, xi, delta, list_y,h):

    pi = np.zeros((t.shape[0]))

    for i in range(t.shape[0]):
        d = h*(t-t[i])**2
        kernel = np.exp(-d)
        pi[i] = np.sum(kernel*xi)/np.sum(kernel)

    sigma = xi*delta

    return np.array(Kaplan_Meier(xi*sigma/pi, t,list_y,False))



def our_estimator(t, xi, delta, list_y,h):

    sigma = xi * delta

    pi = np.zeros((t.shape[0]))
    q = np.zeros((t.shape[0]))

    for i in range(t.shape[0]):
        d = h * (t - t[i]) ** 2
        kernel = np.exp(-d)
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


    return np.array(Kaplan_Meier(mix_delta_p, t,list_y,False))


nb_iter = 100
size = 500

list_measures = list(np.arange(0,5,0.1))

t, c, y, delta, xi, p = generate_data(nb_iter,size)

uncensored_cumulative_hazard_curve = []

for measure in list_measures:
    uncensored_cumulative_hazard_curve.append(expon().cdf(measure))

# kernel bandwith
lambda_ = 0.1

cdf_Subramanian_2004 = np.zeros((len(list_measures)))
cdf_Subramanian_2006 = np.zeros((len(list_measures)))
cdf_our_estimator = np.zeros((len(list_measures)))


for j in range(nb_iter):
    print("iter : " + str(j))
    cdf_Subramanian_2004 += Subramanian_2004(t[j,:],xi[j,:],np.copy(delta[j,:]),list_measures, lambda_)
    cdf_Subramanian_2006 += Subramanian_2006(t[j, :], xi[j, :], np.copy(delta[j, :]), list_measures, lambda_)
    cdf_our_estimator += our_estimator(t[j, :], xi[j, :], np.copy(delta[j, :]), list_measures, lambda_)

cdf_Subramanian_2004 = cdf_Subramanian_2004/nb_iter
cdf_Subramanian_2006 = cdf_Subramanian_2006/nb_iter
cdf_our_estimator = cdf_our_estimator/nb_iter

plt.plot(list_measures,uncensored_cumulative_hazard_curve, label='True cdf')
plt.plot(list_measures,cdf_Subramanian_2004, label='cdf_Subramanian_2004')
plt.plot(list_measures,cdf_Subramanian_2006, label='cdf_Subramanian_2006')
plt.plot(list_measures,cdf_our_estimator, label='cdf_our_estimator')

plt.legend()
plt.show()


#### MISE evaluation ####

uncensored_cumulative_hazard_curve = []


ground_truth_cdf = np.zeros((nb_iter, size))
esimated_cdf_Subramanian_2004 = np.zeros((nb_iter, size))
esimated_cdf_Subramanian_2006 = np.zeros((nb_iter, size))
esimated_cdf_our_estimator = np.zeros((nb_iter, size))


for j in range(nb_iter):
    print("iter : " + str(j))

    list_measures = list(y[j,:])
    list_measures.sort()

    esimated_cdf_Subramanian_2004[j,:] = Subramanian_2004(t[j,:],xi[j,:],np.copy(delta[j,:]),list_measures, lambda_)
    esimated_cdf_Subramanian_2006[j,:] = Subramanian_2006(t[j, :], xi[j, :], np.copy(delta[j, :]), list_measures, lambda_)
    esimated_cdf_our_estimator[j,:] = our_estimator(t[j, :], xi[j, :], np.copy(delta[j, :]), list_measures, lambda_)

    for idx, measure in enumerate(list_measures):
        ground_truth_cdf[j, idx] = expon().cdf(measure)


MISE_Subramanian_2004 = np.sum((esimated_cdf_Subramanian_2004 - ground_truth_cdf)**2)/(nb_iter*size)
MISE_Subramanian_2006 = np.sum((esimated_cdf_Subramanian_2006 - ground_truth_cdf)**2)/(nb_iter*size)
MISE_our_estimator = np.sum((esimated_cdf_our_estimator - ground_truth_cdf)**2)/(nb_iter*size)

print("MISE Subramanian 2004 ")
print(MISE_Subramanian_2004)

print("MISE Subramanian 2006 ")
print(MISE_Subramanian_2006)

print("MISE our estimator")
print(MISE_our_estimator)