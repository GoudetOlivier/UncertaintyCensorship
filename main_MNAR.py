from fn import *
import torch
from os import system
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import argparse
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

from missingCensorshipModels import HeckMan_MAR, HeckMan_MNAR,  HeckMan_MNAR_two_steps, Linear, Neural_network_regression, WeibullMechanism


##########
#  Main  #
##########
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('num_expe', metavar='d', type=float, help='data')
    
    args = parser.parse_args()
    
    num_expe = args.num_expe
    
    
    
    rho = 0.25 * num_expe%4

    #device = "cuda:" + str(int(num_device))
    device = "cpu"

    

    nb_iter = 200

    sample_size = 1000


    name = "MAR001"


    n_jobs_cross_val = 20

    x_list = [0.3, 0.5, 0.7]

    list_h = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


    lr1 = 0.001
    lr2 = 0.001


    nb_epoch_xi = 1000
    nb_epoch_delta_rho = 1000

    batch_size = 100



    for type_data in ["weibull","gamma"]:

        for noCovariateMode in [  False ]:

            for ll in [1]:

                if (type_data == "weibull"):

                    if(noCovariateMode):
                        a = [3, 0, 0, 0]
                        b = [2.5, 0, 0, 0]
                        c = [0.4, 0, -0.9]
                    else:
                        a = [3, 0.5, 0.7,0.2]
                        b = [2.5, -2, 0.4,-0.5]
                        c = [-0.7,1,0.5]

                    Y, C, T, delta, xi, X, XS, probaDelta, probaXi = test_gen_data_weibull_model_Heckman_Mnar(nb_iter, sample_size,a, b, c, rho)

                elif (type_data == "pareto"):

                    if (noCovariateMode):
                        a = [3, 0, 0, 0]
                        b = [0.5, 0, 0, 0]
                        c = [0.3, 0, -0.3]
                    else:
                        a = [3, 3, 4,1]
                        b = [2, -0.1, 0.4,-0.5]
                        c = [-0.2, 1, -0.3]

                    Y, C, T, delta, xi, X, XS, probaDelta, probaXi = test_gen_data_pareto_model_Heckman_Mnar(nb_iter, sample_size,
                                                                                                          a, b, c, rho)
                elif (type_data == "gamma"):

                    if (noCovariateMode):
                        a = [3, 0, 0, 0]
                        b = [0.5, 0, 0, 0]
                        c = [0.3, 0, -0.3]
                    else:
                        a = [0.1, 2, 0.2,1]
                        b = [1, -0.1, 0.4,-0.5]
                        c = [-0.2, 1, -0.3]

                    Y, C, T, delta, xi, X, XS, probaDelta, probaXi = test_gen_data_gamma_model_Heckman_Mnar(nb_iter, sample_size,
                                                                                                          a, b, c, rho)

                print("frac delta obs")
                print(np.sum(xi[0])/sample_size)
                print("th.sum(delta * xi)")
                print(np.sum(delta[0] * xi[0])/sample_size)
                print("th.sum((1-delta) * xi)")
                print(np.sum((1 - delta[0]) * xi[0])/sample_size)
                print("th.sum(delta * (1-xi))")
                print(np.sum(delta[0] * (1-xi[0]))/sample_size)
                print("th.sum((1-delta) * (1-xi))")
                print(np.sum((1 - delta[0]) * (1-xi[0]))/sample_size)


                list_Y_obs = [ Y[i,xi[i,:] == 1]  for i in range(nb_iter) ]
                list_C_obs = [C[i, xi[i, :] == 1] for i in range(nb_iter)]
                list_T_obs = [T[i, xi[i, :] == 1] for i in range(nb_iter)]
                list_delta_obs = [delta[i, xi[i, :] == 1] for i in range(nb_iter)]
                list_X_obs = [X[i, xi[i, :] == 1] for i in range(nb_iter)]
                list_XS_obs = [XS[i, xi[i, :] == 1] for i in range(nb_iter)]
                list_probaDelta_obs = [probaDelta[i, xi[i, :] == 1] for i in range(nb_iter)]
                list_probaXi_obs = [probaXi[i, xi[i, :] == 1] for i in range(nb_iter)]

                if (X[0].ndim == 2):
                    d = X[0].shape[1]
                else:
                    d = 1

                if (XS[0].ndim == 2):
                    dS = XS[0].shape[1]
                else:
                    dS = 1

                if (noCovariateMode):
                    d = 0
                    dS = 0


                dict_p = {}


                # list_model = ["Standard_beran", "True_proba", "Subramanian", "NN_two_steps", "NN_two_steps_with_delta", "NN"  , "NN_with_delta",   "Linear", "Linear_with_delta", "Linear_MAR", "Linear_MAR_with_delta", "NN_MAR","NN_MAR_with_delta" ]


                #list_model = ["Standard_beran", "True_proba", "Subramanian", "NN_together", "NN_two_steps", "NN_two_steps_with_delta", "NN_MAR",  "NN_MAR_with_delta"]

                list_model = ["Standard_beran", "True_proba", "Subramanian","NN_two_steps", "NN_two_steps_with_delta","NN_MAR",  "NN_MAR_with_delta",  ]

                #list_model = ["NN_MAR", "NN_MAR_with_delta"]
                #list_model = ["Linear", "Linear_with_delta", "Linear_MAR", "Linear_MAR_with_delta"]

                for type_model in list_model:

                    p = np.zeros((nb_iter, sample_size))

                    if (type_model == "Standard_beran"):

                        p = delta

                    elif (type_model == "True_proba"):

                        p = probaDelta

                    # elif (type_model == "Subramanian"):
                    #
                    #     for k in range(nb_iter):
                    #         if(noCovariateMode):
                    #             p[k, :] = Subramanian_estimator_no_covariate(T[k,], delta[k,], xi[k,], 0.5)
                    #         else:
                    #             p[k, :] = Subramanian_estimator(X[k,], T[k,], delta[k,], xi[k,], 0.5)

                    elif (type_model == "Linear" or type_model == "NN" or type_model == "NN_two_steps"  or type_model == "sameClass"):

                        if(type_model == "Linear"):

                            f = Linear(d+1)
                            g = Linear(dS+1)



                        elif(type_model == "NN" or type_model == "NN_two_steps" ):


                            if(noCovariateMode):
                                f = Neural_network_regression([d + 1, 200, 200, 100, 1])
                                g = Neural_network_regression([dS + 1, 200, 200, 100, 1])
                                # f = Neural_network_regression([d + 1, 200, 100, 1])
                                # g = Neural_network_regression([dS + 1, 200, 100, 1])
                            else:
                                f = Neural_network_regression([d + 1, 200, 200, 100, 1])
                                g = Neural_network_regression([dS + 1, 200, 200, 100, 1])

                        elif (type_model == "sameClass"):
                            if (type_data == "weibull"):
                                f = WeibullMechanism(d)
                                g = Linear(dS + 1)



                        print("lr1 : " + str(lr1))
                        print("lr2 : " + str(lr2))


                        for k in range(nb_iter):

                            relaunch = True
                            while (relaunch == True):

                                # if(type_model == "NN_two_steps"):

                                hMnar = HeckMan_MNAR_two_steps(f, g, device, noCovariateMode)

                                print("fit xi")
                                hMnar.fit_xi(XS[k,], T[k,], xi[k,], probaXi[k,], lr1, nb_epoch_xi, batch_size)



                                print("fit delta rho")
                                hMnar.fit_delta_rho(X[k,], XS[k,], T[k,], delta[k,], xi[k,], probaDelta[k,], lr1, lr2 ,nb_epoch_delta_rho, batch_size)

                                p[k, :] = hMnar.predict(X[k,], T[k,])

                                if (np.isnan(np.sum(p[k, :])) == False):

                                    relaunch = False


                                # else:
                                #
                                #     print("NN_together")
                                #     hMnar = HeckMan_MNAR(f, g, device, noCovariateMode)
                                #     hMnar.fit(X[k,], XS[k,], T[k,], delta[k,], xi[k,], probaDelta[k,],lr1, lr2 ,nb_epoch_delta_rho,batch_size)



                    elif (type_model == "Linear_MAR" or type_model == "NN_MAR" or type_model == "sameClass_MAR"):


                        if (type_model == "Linear_MAR"):

                            f = Linear(d+1)

                        elif (type_model == "NN_MAR"):

                            if(noCovariateMode):
                                f = Neural_network_regression([d + 1, 200, 200, 100, 1])

                            else:
                                f = Neural_network_regression([d + 1, 200, 200, 100, 1])


                        lr = 0.001
                        

                        for k in range(nb_iter):

                            relaunch = True

                            while(relaunch):

                                mnar = HeckMan_MAR(f,  device, noCovariateMode)

                                mnar.fit(list_X_obs[k],  list_T_obs[k], list_delta_obs[k], list_probaDelta_obs[k], lr, nb_epoch_delta_rho, batch_size)

                                p[k, :] = mnar.predict(X[k,], T[k,])

                                if(np.isnan(np.sum(p[k, :]))==False):
                                    relaunch = False


                    if (type_model == "Linear_with_delta"):
                        dict_p["Linear_with_delta"] = dict_p["Linear"] * (1-xi) + delta * xi
                    elif(type_model == "NN_with_delta"):
                        dict_p["NN_with_delta"] = dict_p["NN"] * (1 - xi) + delta * xi
                    elif(type_model == "Linear_MAR_with_delta"):
                        dict_p["Linear_MAR_with_delta"] = dict_p["Linear_MAR"] * (1 - xi) + delta * xi
                    elif(type_model == "NN_MAR_with_delta"):
                        dict_p["NN_MAR_with_delta"] = dict_p["NN_MAR"] * (1 - xi) + delta * xi

                    elif (type_model == "NN_two_steps_with_delta"):
                        dict_p["NN_two_steps_with_delta"] = dict_p["NN_two_steps"] * (1 - xi) + delta * xi


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


                    if(noCovariateMode == False):
                        if (type_model == "Standard_beran"):
                            list_best_h = Parallel(n_jobs=n_jobs_cross_val)(
                                delayed(cross_val_beran)(T.shape[1], T[k, :], delta[k, :], p[k, :],
                                                         X[k, :], list_h, k) for k in
                                range(nb_iter))

                        elif (type_model == "Subramanian"):

                            print("OK OK ")
                            # cross_val_beran_Subramanian(n, obs, delta, xi, x, list_h1, list_h2, k):

                            list_best_h = Parallel(n_jobs=n_jobs_cross_val)(
                                delayed(cross_val_beran_Subramanian_beran)(T.shape[1], T[k, :], delta[k, :], xi[k, :],
                                                         X[k, :], list_h, list_h, k) for k in
                                range(nb_iter))




                        #     if (noCovariateMode):
                        # #             p[k, :] = Subramanian_estimator_no_covariate(T[k,], delta[k,], xi[k,], 0.5)
                        # #         else:
                        # #             p[k, :] = Subramanian_estimator(X[k,], T[k,], delta[k,], xi[k,], 0.5)
                        #

                        else:
                            list_best_h = Parallel(n_jobs=n_jobs_cross_val)(
                                delayed(cross_val_beran_proba)(T.shape[1], T[k, :], p[k, :], X[k, :],
                                                               list_h, k) for k in
                                range(nb_iter))

                    else:

                        if (type_model == "Subramanian"):

                            list_best_h = Parallel(n_jobs=n_jobs_cross_val)(
                                delayed(cross_val_beran_Subramanian_Kaplan)(T.shape[1], T[k, :], delta[k, :], xi[k, :],
                                                                           X[k, :], list_h,  k) for k in
                                range(nb_iter))

                            print("list_best_h")
                            print(list_best_h)



                    for k in pbar:


                        if (type_model == "Subramanian"):

                            if(noCovariateMode):
                                p[k, :] = Subramanian_estimator_no_covariate(T[k,], delta[k,], xi[k,], list_best_h[k])
                            else:
                                p[k, :] = Subramanian_estimator(X[k,], T[k,], delta[k,], xi[k,],list_best_h[k][0] )

                        c_x = 0
                        for x_eval in x_list:

                            if(noCovariateMode):
                                beran[k, :, c_x] = beran_estimator(t, T[k, :], p[k, :], X[k, :], x_eval, -1, False,
                                                                   True)
                            else:
                                if (type_model == "Subramanian"):
                                    beran[k, :, c_x] = beran_estimator(t, T[k, :], p[k, :], X[k, :], x_eval, list_best_h[k][1],
                                                                       False, False)
                                else:
                                    beran[k, :, c_x] = beran_estimator(t, T[k, :], p[k, :], X[k, :], x_eval, list_best_h[k], False, False)

                            c_x += 1

                    dict_beran[type_model] = beran

                    np.save("save/" + type_model, beran)


                #### Test beran on observed delta
                print("Standard_Beran_delta_obs_only")
                beran = np.zeros((nb_iter, len(t), len(x_list)))

                if (noCovariateMode == False):
                    list_best_h = Parallel(n_jobs=n_jobs_cross_val)(
                        delayed(cross_val_beran)(list_T_obs[k].shape[0], list_T_obs[k], list_delta_obs[k], list_delta_obs[k],
                                                 list_X_obs[k], list_h, k) for k in range(nb_iter))

                for k in pbar:
                    c_x = 0
                    for x_eval in x_list:
                        if (noCovariateMode == False):
                            beran[k, :, c_x] = beran_estimator(t, list_T_obs[k], list_delta_obs[k], list_X_obs[k], x_eval,
                                                              list_best_h[k], False, False)

                        else:
                            beran[k, :, c_x] = beran_estimator(t, list_T_obs[k], list_delta_obs[k], list_X_obs[k], x_eval, -1, False, True)

                        c_x += 1
                dict_beran["Standard_Beran_delta_obs_only"] = beran
                list_model.append("Standard_Beran_delta_obs_only")




                #######################
                # Compute results     #
                #######################

                df_results_mise = pd.DataFrame()

                true_cdf = np.zeros((num_t, len(x_list)))

                df_results_mise["t"] = t

                for i in range(len(x_list)):

                    if (type_data == "weibull"):

                        true_cdf[:, i] =  scipy.stats.weibull_min(a[0] + a[1] * x_list[i] + a[2] * x_list[i] + a[3] * x_list[i]).cdf(t)

                    elif (type_data == "pareto"):

                        true_cdf[:, i] =  scipy.stats.pareto(a[0] + a[1] * x_list[i] + a[2] * x_list[i] + a[3] * x_list[i]).cdf(t)

                    elif (type_data == "gamma"):

                        true_cdf[:, i] =  scipy.stats.gamma(a[0] + a[1] * x_list[i] + a[2] * x_list[i] + a[3] * x_list[i]).cdf(t)


                    df_results_mise["true_cdf_" + str(x_list[i])] = true_cdf[:, i]

                    for idx, type_model in enumerate(list_model):

                        beran = dict_beran[type_model]
                        mean_beran = np.mean(beran[:, :, i], axis=0)

                        df_results_mise[type_model + "_cdf_" + str(x_list[i])] = mean_beran

                        mise_beran = np.mean((beran[:, :, i] - true_cdf[:, i]) ** 2, axis=0)

                        df_results_mise[type_model + "_mise_" + str(x_list[i])] = mise_beran


                df_results_mise.to_csv(
                    "new_results/Mise_" + name  + type_data + "_n_" + str(T.shape[1]) + "_nbIter_" + str(nb_iter) + "_rho_" + str(rho) + "_noCovariateMode_" + str(noCovariateMode) + ".csv")

                df_results_global_score = pd.DataFrame()

                for idx, type_model in enumerate(list_model):
                    beran = dict_beran[type_model]

                    global_score = np.zeros((nb_iter))

                    for iter in range(nb_iter):
                        global_score[iter] = np.mean((beran[iter, :, :] - true_cdf) ** 2)

                    df_results_global_score[type_model] = global_score


                df_results_global_score.to_csv(
                    "new_results/Global_score_" + name + type_data + "_n_" + str(T.shape[1]) + "_nbIter_" + str(nb_iter) + "_rho_" + str(rho) + "_noCovariateMode_" + str(noCovariateMode) + ".csv")

