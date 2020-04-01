from fn import *
import torch
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
from scipy.stats import norm

##########
#  Main  #
##########
sample_size = 2000
nb_iter = 100

x_list = [0.3,0.5,0.7]



# a0 = 3
# a1 = .5
# a2 = .7
# b0 = 1
# b1 = .5
# b2 = .4
#surv, censor, obs, delta,  xi, x, proba = gen_data_exponential(nb_iter,sample_size, a0, a1, a2, b0, b1, b2)
#

#surv, censor, obs, delta,  xi, x = gen_data_weibull(nb_iter,sample_size)


surv, censor, obs, delta,  xi, x = gen_data_multivariate_model(nb_iter,sample_size)

print(obs.shape)

print(x.shape)

if(x.ndim==2):
	X = np.stack((obs, x),2)
else:
	X = np.concatenate((np.expand_dims(obs, axis=2), x), axis=2)

print("delta")
print(delta[0])

y = delta

dict_p = {}

#list_model = ["Standard_beran","NN_BCE","NN_MSEloss","NN_L1loss","DecisionTree","KNN","LogisticRegression"]

#
#list_model = ["Standard_beran","NN_BCE","NN_MSEloss","DecisionTree","KNN","LogisticRegression"]

list_model = ["Standard_beran"]

list_color = ["blue","red","green","orange","cyan","brown","magenta"]



for type_model in list_model:

	print(type_model)

	p = np.zeros((nb_iter, sample_size))

	if (type_model == "Standard_beran"):

		p = delta

	# elif (type_model == "True_proba"):
    #
	# 	p = proba

	elif(type_model == "NN_BCE" or type_model == "NN_L1loss" or type_model == "NN_MSEloss"):

		nb_epoch = 1000
		layers_size = [X.shape[2], 200, 200, 1]
		isParallel_run = True

		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		#device = "cpu"
		if(type_model == "NN_BCE"):
			nnet = nnetwrapper(nb_iter, layers_size, device, isParallel_run, "BCEloss")
		elif(type_model == "NN_L1loss"):
			nnet = nnetwrapper(nb_iter, layers_size, device, isParallel_run, "L1loss")
		elif (type_model == "NN_MSEloss"):
			nnet = nnetwrapper(nb_iter, layers_size, device, isParallel_run, "MSEloss")

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
				clf = tree.DecisionTreeClassifier(min_samples_split=100)
			elif(type_model =="KNN"):
				n_neighbors = 10
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


list_h = [0.1,0.2,0.5,0.8,1]


for type_model in list_model:

	print(type_model)

	beran = np.zeros((nb_iter,len(t),len(x_list)))

	pbar = tqdm(range(nb_iter))

	p = dict_p[type_model]

	for k in pbar:

		h_best = cross_val_beran(sample_size, obs[k, :], delta[k,:], p[k, :], x[k, :], list_h )

		print("best h : " + str(h_best))

		# Bandwidth selection
		h = 0.5
		c_x = 0
		for x_eval in x_list:
			# Estimators computation

			beran[k, :, c_x] = gene_Beran(t, obs[k, :], p[k, :], x[k, :], x_eval, h)

			#beran[k, :, c_x] = beran_estimator(p[k, :],t,obs[k, :],x[k, :],x_eval,h,False)


			c_x += 1

	dict_beran[type_model] = beran

	np.save("save/" + type_model, beran)



################
# Plot results #
################
plt.figure()


for i in range(len(x_list)):

	#true_cdf = expon(scale=1/(a0+a1*x_list[i]+a2*x_list[i]**2)).cdf(t)
	#true_cdf = exponweib.cdf(t, 1, 0.5*(x_list[i] + 4))

	#true_cdf = expon(scale=(0.1*x_list[i] + 0.2 *x_list[i] +  0.3 *x_list[i] +  0.4 *x_list[i] + 0.5 *x_list[i])).cdf(t)

	# true_cdf = expon(
	# 	scale=(5 + 1/5*(np.sin(x_list[i])+ np.cos(x_list[i]) + x_list[i]**2 + np.exp(x_list[i]) + x_list[i]))).cdf(t)

	true_cdf = norm(loc=5 + 1/5*(np.sin(x_list[i])+ np.cos(x_list[i]) + x_list[i]**2 + np.exp(x_list[i]) + x_list[i]),scale = 0.3).cdf(t)

	plt.subplot(len(x_list), 2, 2 * i + 1 )
	plt.plot(t, true_cdf, label='cdf',color="black")

	for idx, type_model in enumerate(list_model):

		beran = dict_beran[type_model]
		mean_beran = np.mean(beran[:,:,i],axis=0)
		plt.plot(t,mean_beran,label=type_model, color= list_color[idx])

	plt.legend()

	plt.subplot(len(x_list),2,2*(i+1))

	for idx, type_model in enumerate(list_model):

		beran = dict_beran[type_model]
		mise_beran = np.mean((beran[:, :, i] - true_cdf) ** 2, axis=0)
		plt.plot(t,mise_beran,label=type_model,color= list_color[idx])

	plt.legend()

plt.show()