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

from missingCensorshipModels import MAR, HeckMan_MNAR,  Linear, Neural_network_regression


##########
#  Main  #
##########
if __name__ == '__main__':

    nb_iter = 100

    sample_size = 1000




    n_jobs_cross_val = 20

    x_list = [0.3, 0.5, 0.7]

    list_h = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]




    for type_data in ["multivariate"]:

        if (type_data == "exponential"):

            a0 = 1
            b0 = 1
            c0 = 2

            rho = 0.5

            Y, C, T, delta, xi,  X, XS, probaDelta = test_gen_data_exponential_Heckman_Mnar(nb_iter,sample_size, a0,
                                                                                                          b0, c0, rho)
            print("np.sum(xi[0])/sample_size")
            print(np.sum(xi[0])/sample_size)

        elif (type_data == "multivariate"):

                rho = 0.5

                Y, C, T, delta, xi,  X, XS, probaDelta = test_gen_data_multivariate_model_Heckman_Mnar(nb_iter, sample_size, rho)

                print("np.sum(xi[0])/sample_size")
                print(np.sum(xi[0]) / sample_size)


                list_Y_obs = [ Y[i,xi[i,:] == 1]  for i in range(nb_iter) ]
                list_C_obs = [C[i, xi[i, :] == 1] for i in range(nb_iter)]
                list_T_obs = [T[i, xi[i, :] == 1] for i in range(nb_iter)]
                list_delta_obs = [delta[i, xi[i, :] == 1] for i in range(nb_iter)]
                list_X_obs = [X[i, xi[i, :] == 1] for i in range(nb_iter)]
                list_XS_obs = [XS[i, xi[i, :] == 1] for i in range(nb_iter)]
                list_probaDelta_obs = [probaDelta[i, xi[i, :] == 1] for i in range(nb_iter)]


        if (X[0].ndim == 2):
            d = X[0].shape[1]
        else:
            d = 1

        if (XS[0].ndim == 2):
            dS = XS[0].shape[1]
        else:
            dS = 1


        dict_p = {}

        # list_model = ["Standard_beran", "True_proba", ]

        list_model = ["Standard_beran", "True_proba", "NN" , "NN_with_delta",  "Linear", "Linear_with_delta",  "Linear_MAR", "NN_MAR" ]

        for type_model in list_model:

            p = np.zeros((nb_iter, sample_size))

            if (type_model == "Standard_beran"):

                p = delta

            elif (type_model == "True_proba"):

                p = probaDelta

            elif (type_model == "Linear"):

                print("d : " + str(d))
                print("dS : " + str(dS))

                f = Linear(d+1)
                g = Linear(dS+1)

                for k in range(nb_iter):

                    hMnar = HeckMan_MNAR(f, g, "cpu")


                    hMnar.fit(X[k,], XS[k,], T[k,], delta[k,], xi[k,], probaDelta[k,],0.00005, 0.001 ,200,100)

                    p[k, :] = hMnar.predict(X[k,], T[k,])




            elif (type_model == "NN"):

                f = Neural_network_regression([d+1, 200, 100, 1])
                g = Neural_network_regression([dS+1, 200, 100, 1])


                for k in range(nb_iter):

                    hMnar = HeckMan_MNAR(f, g, "cpu")


                    hMnar.fit(X[k], XS[k], T[k], delta[k], xi[k], probaDelta[k], 0.00005, 0.001, 200,100)

                    p[k, :] = hMnar.predict(X[k,], T[k,])


            elif (type_model == "Linear_MAR"):

                f = Linear(d+1)

                for k in range(nb_iter):

                    mnar = MAR(f,  "cpu")

                    mnar.fit(list_X_obs[k],  list_T_obs[k], list_delta_obs[k], list_probaDelta_obs[k], 0.001, 1000,100)

                    p[k, :] = mnar.predict(X[k,], T[k,])


            elif (type_model == "NN_MAR"):

                f = Neural_network_regression([d + 1, 200, 100, 1])

                for k in range(nb_iter):

                    mnar = MAR(f,  "cpu")

                    mnar.fit(list_X_obs[k],  list_T_obs[k], list_delta_obs[k], list_probaDelta_obs[k], 0.001, 1000,100)

                    p[k, :] = mnar.predict(X[k,], T[k,])




            if (type_model == "Linear_with_delta"):

                dict_p["Linear_with_delta"] = dict_p["Linear"] * (1-xi) + delta * xi

            elif(type_model == "NN_with_delta"):

                dict_p["NN_with_delta"] = dict_p["NN"] * (1 - xi) + delta * xi

            else:

                dict_p[type_model] = p


        #######################
        # Survival estimation #
        #######################
        print('Survival estimators computing')

        dict_beran = {}

        num_t = 100

        t = np.linspace(np.amin(Y), np.amax(Y), num=num_t)

        for type_model in list_model:

            print(type_model)

            beran = np.zeros((nb_iter, len(t), len(x_list)))

            pbar = tqdm(range(nb_iter))

            p = dict_p[type_model]

            # list_best_h = Parallel(n_jobs=n_jobs_cross_val)(
            #     delayed(cross_val_beran)(T.shape[1], T[k, :], delta[k, :], p[k, :], X[k, :], list_h, k) for k in
            #     range(nb_iter))

            if (type_model == "Standard_beran"):
                list_best_h = Parallel(n_jobs=n_jobs_cross_val)(
                    delayed(cross_val_beran)(T.shape[1], T[k, :], delta[k, :], p[k, :],
                                             X[k, :], list_h, k) for k in
                    range(nb_iter))

            else:
                list_best_h = Parallel(n_jobs=n_jobs_cross_val)(
                    delayed(cross_val_beran_proba)(T.shape[1], T[k, :], p[k, :], X[k, :],
                                                   list_h, k) for k in
                    range(nb_iter))

            for k in pbar:

                c_x = 0
                for x_eval in x_list:

                    beran[k, :, c_x] = beran_estimator(t, T[k, :], p[k, :], X[k, :], x_eval, list_best_h[k], False)

                    c_x += 1

            dict_beran[type_model] = beran

            np.save("save/" + type_model, beran)


        #### Test beran on observed delta
        print("Standard_Beran_delta_obs_only")
        beran = np.zeros((nb_iter, len(t), len(x_list)))
        list_best_h = Parallel(n_jobs=n_jobs_cross_val)(
            delayed(cross_val_beran)(list_T_obs[k].shape[0], list_T_obs[k], list_delta_obs[k], list_delta_obs[k],
                                     list_X_obs[k], list_h, k) for k in range(nb_iter))

        for k in pbar:
            c_x = 0
            for x_eval in x_list:
                beran[k, :, c_x] = beran_estimator(t, list_T_obs[k], list_delta_obs[k], list_X_obs[k], x_eval, list_best_h[k], False)

                c_x += 1
        dict_beran["Standard_Beran_delta_obs_only"] = beran
        list_model.append("Standard_Beran_delta_obs_only")

        # plt.figure()

        df_results_mise = pd.DataFrame()

        true_cdf = np.zeros((num_t, len(x_list)))

        df_results_mise["t"] = t

        for i in range(len(x_list)):

            if (type_data == "exponential"):

                true_cdf[:, i] = expon(scale=1 / (a0 * x_list[i])).cdf(t)

            elif (type_data == "multivariate"):

                true_cdf[:, i] = norm(loc=0.5 + 1 / 5 * (
                        np.sin(x_list[i]) + np.cos(x_list[i]) + x_list[i] ** 2 - 0.1 * np.exp(x_list[i]) + x_list[i]),
                                      scale=0.3).cdf(t)



            df_results_mise["true_cdf_" + str(x_list[i])] = true_cdf[:, i]

            for idx, type_model in enumerate(list_model):

                beran = dict_beran[type_model]
                mean_beran = np.mean(beran[:, :, i], axis=0)

                df_results_mise[type_model + "_cdf_" + str(x_list[i])] = mean_beran

                mise_beran = np.mean((beran[:, :, i] - true_cdf[:, i]) ** 2, axis=0)

                df_results_mise[type_model + "_mise_" + str(x_list[i])] = mise_beran


        df_results_mise.to_csv(
            "results/Mise_" + type_data + "_n_" + str(T.shape[1]) + "_nbIter_" + str(nb_iter) + ".csv")

        df_results_global_score = pd.DataFrame()

        for idx, type_model in enumerate(list_model):
            beran = dict_beran[type_model]

            global_score = np.zeros((nb_iter))

            for iter in range(nb_iter):
                global_score[iter] = np.mean((beran[iter, :, :] - true_cdf) ** 2)

            df_results_global_score[type_model] = global_score


        df_results_global_score.to_csv(
            "results/Global_score_" + type_data + "_n_" + str(T.shape[1]) + "_nbIter_" + str(nb_iter) + ".csv")

        # plt.savefig("results/fig_new2_" + type_data + "_n_" + str(T.shape[1]) + "_nbIter_" + str(nb_iter) + ".png")
