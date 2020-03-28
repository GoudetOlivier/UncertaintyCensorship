from fn import *
# import torch
from os import system
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

from neuralnets import NNetWrapper as nnetwrapper
from sklearn import neighbors
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from scipy.stats import exponweib


##########
#  Main  #
##########
sample_size = 1000
nb_iter = 100

x_list = [.3,.5,.7]



a0 = 3
a1 = .5
a2 = .7
b0 = 1
b1 = .5
b2 = .4
#surv, censor, obs, delta,  xi, x, proba = gen_data_exponential(nb_iter,sample_size, a0, a1, a2, b0, b1, b2)

surv, censor, obs, delta,  xi, x = gen_data_weibull(nb_iter,sample_size)

X = np.stack((obs, x),2)
y = delta

dict_p = {}

#list_model = ["Standard_beran","NN","DecisionTree","KNN","LogisticRegression"]
list_model = ["Standard_beran","DecisionTree","LogisticRegression"]

for type_model in list_model:

	print(type_model)

	p = np.zeros((nb_iter, sample_size))

	if (type_model == "Standard_beran"):

		p = delta

	# elif (type_model == "True_proba"):
    #
	# 	p = proba

	elif(type_model == "NN"):

		nb_epoch = 1000
		layers_size = [2, 200, 1]
		isParallel_run = True

		# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		device = "cpu"

		nnet = nnetwrapper(nb_iter, layers_size, device, isParallel_run)

		if (isParallel_run):
			nnet.fit(nb_epoch, X, y)
			p = nnet.predict(X)
		else:
			for k in range(nb_iter):
				print("iter : " + str(k))

				nnet.fit(nb_epoch, X[k,], y[k,])
				p[k, :] = nnet.predict(X[k,])

	else:
		for k in range(nb_iter):

			if(type_model == "DecisionTree"):
				clf = tree.DecisionTreeClassifier(min_samples_split=50)
			elif(type_model =="KNN"):
				n_neighbors = 300
				clf = neighbors.KNeighborsClassifier(n_neighbors)
			elif(type_model =="LogisticRegression"):
				clf = LogisticRegression(random_state=0)


			clf.fit(X[k,], y[k,])

			p[k, :] = clf.predict_proba(X[k,])[:,1]

	dict_p[type_model] = p


#######################
# Survival estimation #
#######################
print('Survival estimators computing')


dict_beran = {}


t = np.linspace(np.amin(surv),np.amax(surv),num=100)


for type_model in list_model:

	print(type_model)

	beran = np.zeros((nb_iter,len(t),len(x_list)))

	pbar = tqdm(range(nb_iter))

	p = dict_p[type_model]

	for k in pbar:
		# Bandwidth selection
		h = .1
		c_x = 0
		for x_eval in x_list:
			# Estimators computation

			beran[k, :, c_x] = gene_Beran(t, obs[k, :], p[k, :], x[k, :], x_eval, h)

			# beran[k, :, c_x] = Beran_estimator(p[k, :],t,obs[k, :],x[k, :],x_eval,h,False)



			c_x += 1

	dict_beran[type_model] = beran

	np.save("save/" + type_model, beran)



################
# Plot results #
################
plt.figure()

for i in range(len(x_list)):

	#true_cdf = expon(scale=1/(a0+a1*x_list[i]+a2*x_list[i]**2)).cdf(t)
	true_cdf = exponweib.cdf(t, 1, 0.5*(x_list[i] + 4))

	plt.subplot(len(x_list), 2, 2 * i + 1 )
	plt.plot(t, true_cdf, label='cdf')

	for type_model in list_model:

		beran = dict_beran[type_model]
		mean_beran = np.mean(beran[:,:,i],axis=0)
		plt.plot(t,mean_beran,label=type_model)

	plt.legend()

	plt.subplot(len(x_list),2,2*(i+1))

	for type_model in list_model:

		beran = dict_beran[type_model]
		mise_beran = np.mean((beran[:, :, i] - true_cdf) ** 2, axis=0)
		plt.plot(t,mise_beran,label=type_model)

	plt.legend()

plt.show()