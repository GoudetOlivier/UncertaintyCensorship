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

from sklearn.ensemble import RandomForestClassifier

from joblib import Parallel, delayed
import warnings

import pandas as pd

##########
#  Main  #
##########
if __name__ == '__main__':

    sample_size = 2000

    nb_iter = 1000

    n_jobs_cross_val = 20

    x_list = [0.3, 0.5, 0.7]

    list_h = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    test_mode = True

    for type_data in ["exponential"]:

        if (type_data == "exponential"):

            a0 = 3
            a1 = .5
            a2 = .7
            b0 = 1
            b1 = .5
            b2 = .4
            surv, censor, obs, delta, xi, x, proba = gen_data_exponential(nb_iter, sample_size, a0, a1, a2, b0, b1, b2)


            if(test_mode):
                surv_test, censor_test, obs_test, delta_test, xi_test, x_test, proba_test = gen_data_exponential(nb_iter, sample_size, a0, a1, a2, b0, b1,  b2)
               
        elif (type_data == "multivariate"):

            surv, censor, obs, delta, xi, x, proba = gen_data_multivariate_model(nb_iter, sample_size)

            if (test_mode):
                surv_test, censor_test, obs_test, delta_test, xi_test, x_test, proba_test =  gen_data_multivariate_model(nb_iter, sample_size)



        if (x.ndim == 2):
            X = np.stack((obs, x), 2)
        else:
            X = np.concatenate((np.expand_dims(obs, axis=2), x), axis=2)

        if (test_mode):
            if (x_test.ndim == 2):
                X_test = np.stack((obs_test, x_test), 2)
            else:
                X_test = np.concatenate((np.expand_dims(obs_test, axis=2), x_test), axis=2)


        y = delta

        dict_p = {}

        list_model = ["Standard_beran","True_proba", "Prior_proba", "LogisticRegression","NN_BCE"]

        #list_model = ["Standard_beran", "True_proba", "Prior_proba", "LogisticRegression", "NN_BCE", "RandomForest",
        # 			  "KNN"]

        #list_model = ["Standard_beran", "LogisticRegression"]


        #list_model = ["Standard_beran", "True_proba", "Prior_proba", "LogisticRegression"]


        for type_model in list_model:

            print(type_model)

            p = np.zeros((nb_iter, obs.shape[1]))

            if (test_mode):
                p_test = np.zeros((nb_iter, obs.shape[1]))

            if (type_model == "Standard_beran"):

                p = delta
                if (test_mode):
                    p_test = delta_test

            elif (type_model == "True_proba"):

                p = proba

                if (test_mode):
                    p_test = proba_test

            elif (type_model == "Prior_proba"):

                p = proba ** 2 + delta * (1 - proba) + 0.01*np.random.rand(nb_iter,sample_size)

                if (test_mode):
                    p_test = proba_test ** 2 + delta_test * (1 - proba_test) + 0.01*np.random.rand(nb_iter,sample_size)

            elif (type_model == "NN_BCE" or type_model == "NN_L1loss" or type_model == "NN_MSEloss"):

                nb_epoch = 200

                layers_size = [X.shape[2], 100, 100, 1]
                isParallel_run = True

                #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                device = "cpu"

                if (type_model == "NN_BCE"):
                    nnet = nnetwrapper(nb_iter, layers_size, device, isParallel_run, "BCEloss")
                elif (type_model == "NN_L1loss"):
                    nnet = nnetwrapper(nb_iter, layers_size, device, isParallel_run, "L1loss")
                elif (type_model == "NN_MSEloss"):
                    nnet = nnetwrapper(nb_iter, layers_size, device, isParallel_run, "MSEloss")

                if (isParallel_run):

                    nnet.fit(nb_epoch, X, y)
                    p = nnet.predict(X)

                    if (test_mode):
                        p_test = nnet.predict(X_test)

                else:
                    for k in range(nb_iter):
                        print("iter : " + str(k))

                        nnet.fit(nb_epoch, X[k,], y[k,])
                        p[k, :] = nnet.predict(X[k,])

                        if (test_mode):
                            p_test[k, :] = nnet.predict(X_test[k,])


            else:
                for k in range(nb_iter):

                    if (type_model == "RandomForest"):

                        clf = RandomForestClassifier(n_estimators=1000)

                    elif (type_model == "KNN"):
                        n_neighbors = 10
                        clf = neighbors.KNeighborsClassifier(n_neighbors)

                    elif (type_model == "LogisticRegression"):

                        clf = LogisticRegression(random_state=0)

                    clf.fit(X[k,], y[k,])

                    p[k, :] = clf.predict_proba(X[k,])[:, 1]

                    if (test_mode):

                        p_test[k, :] = clf.predict_proba(X_test[k,])[:, 1]

            dict_p[type_model] = p

            if (test_mode):
                dict_p[type_model + "_test_mode"] = p_test




        #######################
        # Survival estimation #
        #######################
        print('Survival estimators computing')

        dict_beran = {}

        num_t = 100

        t = np.linspace(np.amin(surv), np.amax(surv), num=num_t)

        for type_model in list_model:

            print(type_model)

            beran = np.zeros((nb_iter, len(t), len(x_list)))

            pbar = tqdm(range(nb_iter))


            p = dict_p[type_model]


            list_best_h = Parallel(n_jobs=n_jobs_cross_val)(
                delayed(cross_val_beran)(obs.shape[1], obs[k, :], delta[k, :], p[k, :], x[k, :], list_h, k) for k in
               range(nb_iter))



            for k in pbar:

                c_x = 0
                for x_eval in x_list:
                    # beran[k, :, c_x] = gene_Beran(t, obs[k, :], p[k, :], x[k, :], x_eval, h_best)
                    beran[k, :, c_x] = beran_estimator(t, obs[k, :], p[k, :], x[k, :], x_eval, list_best_h[k], False)

                    c_x += 1

            dict_beran[type_model] = beran

            np.save("save/" + type_model, beran)

            if (test_mode):

                p_test = dict_p[type_model + "_test_mode"]

                beran = np.zeros((nb_iter, len(t), len(x_list)))

                if(type_model == "Standard_beran"):
                    list_best_h = Parallel(n_jobs=n_jobs_cross_val)(
                delayed(cross_val_beran)(obs_test.shape[1], obs_test[k, :], delta_test[k, :], p_test[k, :], x_test[k, :], list_h, k) for k in
               range(nb_iter))

                else:
                    list_best_h = Parallel(n_jobs=n_jobs_cross_val)(
                     delayed(cross_val_beran_proba)(obs_test.shape[1], obs_test[k, :],  p_test[k, :], x_test[k, :], list_h, k) for k in
                     range(nb_iter))


                for k in pbar:

                    c_x = 0
                    for x_eval in x_list:
                    
                        beran[k, :, c_x] = beran_estimator(t, obs_test[k, :], p_test[k, :], x_test[k, :], x_eval, list_best_h[k],
                                                           False)

                        c_x += 1

                dict_beran[type_model + "_test_mode"] = beran


                np.save("save/" + type_model + "_test_mode", beran)



        plt.figure()

        df_results_mise = pd.DataFrame()

        true_cdf = np.zeros((num_t, len(x_list)))

        df_results_mise["t"] = t

        for i in range(len(x_list)):

            if (type_data == "exponential"):

                true_cdf[:, i] = expon(scale=1 / (a0 + a1 * x_list[i] + a2 * x_list[i] ** 2)).cdf(t)

            elif (type_data == "multivariate"):

                true_cdf[:, i] = norm(loc=1 + 1 / 5 * (
                np.sin(x_list[i]) + np.cos(x_list[i]) + x_list[i] ** 2 + np.exp(x_list[i]) + x_list[i]), scale=0.3).cdf(
                    t)


            # plt.subplot(len(x_list), 2, 2 * i + 1)
            # plt.plot(t, true_cdf[:, i], label='cdf', color="black")

            df_results_mise["true_cdf_" + str(x_list[i])] = true_cdf[:, i]

            for idx, type_model in enumerate(list_model):
                beran = dict_beran[type_model]
                mean_beran = np.mean(beran[:, :, i], axis=0)

                df_results_mise[type_model + "_cdf_" + str(x_list[i])] = mean_beran

                mise_beran = np.mean((beran[:, :, i] - true_cdf[:, i]) ** 2, axis=0)

                df_results_mise[type_model + "_mise_" + str(x_list[i])] = mise_beran


                if (test_mode):

                    beran = dict_beran[type_model + "_test_mode"]
                    mean_beran = np.mean(beran[:, :, i], axis=0)
                    df_results_mise[type_model + "_test_mode" + "_cdf_" + str(x_list[i])] = mean_beran
                    mise_beran = np.mean((beran[:, :, i] - true_cdf[:, i]) ** 2, axis=0)
                    df_results_mise[type_model + "_test_mode" + "_mise_" + str(x_list[i])] = mise_beran


        df_results_mise.to_csv("results/Mise_" + type_data + "_n_" + str(obs.shape[1]) + "_nbIter_" + str(nb_iter) + ".csv")

        df_results_global_score = pd.DataFrame()

        for idx, type_model in enumerate(list_model):
            beran = dict_beran[type_model]

            global_score = np.zeros((nb_iter))

            for iter in range(nb_iter):
                global_score[iter] = np.mean((beran[iter, :, :] - true_cdf) ** 2)

            df_results_global_score[type_model] = global_score

            if (test_mode):

                beran = dict_beran[type_model + "_test_mode"]
                global_score = np.zeros((nb_iter))

                for iter in range(nb_iter):
                    global_score[iter] = np.mean((beran[iter, :, :] - true_cdf) ** 2)

                df_results_global_score[type_model + "_test_mode"] = global_score


        df_results_global_score.to_csv(
            "results/Global_score_" + type_data + "_n_" + str(obs.shape[1]) + "_nbIter_" + str(nb_iter) + ".csv")

        # plt.savefig("results/fig_new2_" + type_data + "_n_" + str(obs.shape[1]) + "_nbIter_" + str(nb_iter) + ".png")
