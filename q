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

    nb_iter = 3

    list_h = [0.1,1,2,2.5,2.8,3,3.2,3.5,4,5]
    #list_h = [2.5,2.8,3,3.2,3.5]

    split_train_test_ratio = 0.5


    # Load train mgus2 data
    data = pd.read_csv("data/data_mgus2.csv")
    obs = data["futime"].values*30.42

    delta = data["death"].values
    # x2 = data2[["age","sex","creat","hgb","mspike"]].replace("F", 1).replace("M", 0)
    x = data[["age", "creat", "hgb", "mspike"]]

    x = x.fillna(value=x.mean()).values

    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x = (x - x_mean) / x_std

    obs_mean = np.mean(obs, axis=0)
    obs_std = np.std(obs, axis=0)
    obs = (obs - obs_mean)/obs_std

    array_test1 = np.quantile(x,0.5, axis=0)
    #array_test1[0] = 63

    array_test2 = np.quantile(x,0.5, axis=0)
    #array_test2[0] = 72

    array_test3 = np.quantile(x,0.5, axis=0)
    #array_test3[0] = 79

    x_list = [array_test1, array_test2,array_test3]

    #x_list = [np.quantile(x,0.25, axis=0),np.quantile(x,0.5, axis=0),np.quantile(x,0.75, axis=0)]

    num_t = 100
    t = np.linspace(np.amin(obs), np.amax(obs), num=num_t)

    list_model = ["Standard_beran", "LogisticRegression", "NN_BCE"]
    #list_model = ["NN_BCE"]


    list_color = ["blue", "red", "green", "orange", "cyan", "brown", "magenta"]

    dict_beran_test = {}

    for iter in range(nb_iter):

        # Split train/test

        indices = np.random.permutation(x.shape[0])
        training_idx, test_idx = indices[:int(x.shape[0] * split_train_test_ratio)], indices[int(
            x.shape[0] * split_train_test_ratio):]

        x_train = x[training_idx, :]
        x_test = x[test_idx, :]

        obs_train = obs[training_idx]
        obs_test = obs[test_idx]

        delta_train = delta[training_idx]
        delta_test = delta[test_idx]

        # Sort train and test data

        # index = np.argsort(obs_train, axis=0)
        # obs_train = obs_train[index]
        # delta_train = delta_train[index]
        # x_train = x_train[index]

        X_train = np.concatenate((np.expand_dims(obs_train, axis=1), x_train), axis=1)
        y_train = delta_train

        index = np.argsort(obs_test, axis=0)
        obs_test = obs_test[index]
        delta_test = delta_test[index]
        x_test = x_test[index]
        X_test = np.concatenate((np.expand_dims(obs_test, axis=1), x_test), axis=1)
        y_test = delta_test


        dict_p_test = {}


        for type_model in list_model:

            print(type_model)

            p_test = np.zeros((X_test.shape[0]))

            if (type_model == "Standard_beran"):


                p_test = delta_test

            elif (type_model == "NN_BCE" or type_model == "NN_L1loss" or type_model == "NN_MSEloss"):

                nb_epoch = 200

                layers_size = [X_train.shape[1], 100, 100, 1]
                isParallel_run = False

                #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                device = "cpu"

                nnet = nnetwrapper(1, layers_size, device, False, "BCEloss")

                nnet.fit(nb_epoch, X_train, y_train)
                p_test = nnet.predict(X_test)

            else:

                if (type_model == "RandomForest"):

                    clf = RandomForestClassifier(n_estimators=1000)

                elif (type_model == "KNN"):
                    n_neighbors = 10
                    clf = neighbors.KNeighborsClassifier(n_neighbors)

                elif (type_model == "LogisticRegression"):

                    clf = LogisticRegression(random_state=0)

                clf.fit(X_train, y_train)

                p_test = clf.predict_proba(X_test)[:, 1]


            dict_p_test[type_model] = p_test




        #######################
        # Survival estimation #
        #######################
        print('Survival estimators computing')




        for type_model in list_model:

            print(type_model)

            ############### data _test  ################
            beran_test = np.zeros((len(t), len(x_list)))

            p_test = dict_p_test[type_model]

            if(type_model == "Standard_beran"):
                list_best_h_test = cross_val_beran(X_test.shape[0], np.expand_dims(obs_test, axis=1), np.expand_dims(delta_test, axis=1), np.expand_dims(p_test, axis=1), x_test, list_h, 0)
            else:
                list_best_h_test = cross_val_beran_proba(X_test.shape[0], np.expand_dims(obs_test, axis=1), np.expand_dims(p_test, axis=1), x_test, list_h, 0)

            print("list_best_h_test")
            print(list_best_h_test)

            c_x = 0
            for x_eval in x_list:

                beran_test[:, c_x] = beran_estimator(t, np.expand_dims(obs_test, axis=1), np.expand_dims(p_test, axis=1), x_test, x_eval, list_best_h_test, False)

                c_x += 1

            if(iter == 0):
                dict_beran_test[type_model] = beran_test
            else:
                dict_beran_test[type_model] += beran_test


    for type_model in list_model:
        dict_beran_test[type_model] /= nb_iter

    # Plot and save results mgus
    df_results_mise = pd.DataFrame()
    plt.figure()

    df_results_mise["t"] = t * obs_std + obs_mean

    for i in range(len(x_list)):

        plt.subplot(len(x_list), 1, i+1)

        for idx, type_model in enumerate(list_model):
            beran = dict_beran_test[type_model]
            plt.plot(t, 1-beran[ :, i], label=type_model, color=list_color[idx])

            df_results_mise[type_model + "_cdf_" + str(i)] = beran[ :, i]
            df_results_mise[type_model + "_survival_function_" + str(i)] = 1-beran[ :, i]

        plt.legend()

    df_results_mise.to_csv(
        "results/Cdf_mgus2_test_n_" + str(X_train.shape[0]) + ".csv")

    #plt.show()
    plt.savefig("results/fig_mgus2_test_n_" + str(X_train.shape[0]) + ".png")

