import pandas as pd

import numpy as np

import plotly.express as px

import matplotlib.pyplot as plt
import plotly.graph_objects as go

rho = "0"
x = "0.5"

list_model = ["Standard_beran", "NN_MAR_with_delta", "NN_two_steps_with_delta", "Subramanian",   "True_proba" ]
list_name = ["Standard", "MAR - Neural Network", "MNAR - Neural Network", "MAR - Subramanian type",   "Oracle" ]


dataPath = "save_results/weibull/Mise_weibull_n_1000_nbIter_200_rho_" + rho + "_noCovariateMode_False.csv"

df = pd.read_csv(dataPath)

dataPathLinear = "save_results/weibull/linear/Mise_weibull_n_1000_nbIter_20_rho_" + rho + "_noCovariateMode_False.csv"

list_model_linear = ["Linear_with_delta", "Linear_MAR_with_delta"]
list_name_linear = ["MNAR - Linear model", "MAR - Linear model"]

df_linear = pd.read_csv(dataPathLinear)


# for model in list_model:
#
#     plt.plot(df["t"],df[model + "_cdf_" + x], label=model)
#
# plt.legend()
# plt.show()


# for model in list_model:
#
#     plt.plot(df["t"],df[model + "_mise_" + x], label=model)
#
# plt.legend()
# plt.show()


fig = go.Figure()

fig.add_trace(go.Scatter(x=df["t"].values, y=df["True_proba" + "_mise_" + x].values,
                    line=dict(color='royalblue', width=3, dash='dot'),
                    name="Oracle"))

fig.add_trace(go.Scatter(x=df["t"].values, y=df["Standard_beran" + "_mise_" + x].values,
                    line=dict(color='orange', width=3, dash='dot'),
                    name="Standard"))

fig.add_trace(go.Scatter(x=df["t"].values, y=df["Subramanian" + "_mise_" + x].values,
                    line=dict(color='pink', width=3, dash='dash'),
                    name="MAR - Subramanian type"))

# fig.add_trace(go.Scatter(x=df["t"].values, y=df_linear["Linear_MAR_with_delta" + "_mise_" + x].values,
#                     line=dict( width=3 , dash='dash'),
#                     name="MAR - Linear model"))

fig.add_trace(go.Scatter(x=df["t"].values, y=df["NN_MAR_with_delta" + "_mise_" + x].values,
                    line=dict(color='red', width=3 , dash='dash'),
                    name="MAR - Neural Network"))

# fig.add_trace(go.Scatter(x=df["t"].values, y=df_linear["Linear_with_delta" + "_mise_" + x].values,
#                     line=dict( width=3),
#                     name="MNAR - Linear model"))

fig.add_trace(go.Scatter(x=df["t"].values, y=df["NN_two_steps_with_delta" + "_mise_" + x].values,
                    line=dict(color='green', width=3),
                    name="MNAR - Neural Network"))


fig.update_layout(
    xaxis_title=r'$\Large{y_l}$',
    #xaxis_title=r'$\sqrt{(n_\text{c}(t|{T_\text{early}}))}$',
    yaxis_title=r'$\Large{\text{MSE}}$',
    title={'text' : r'$\Huge{\rho=' + rho + '}$',
            'x':0.4,
            'xanchor': 'center',
            'yanchor': 'top'
        },
    legend_title="Beran estimators:",
    font=dict(
        family="Courier New, monospace",
        size=24
    )
)


fig.show()



# total = 0
#
# model = "True_proba"
#
# for x in ["0.3","0.5","0.7"]:
#     total += df[model + "_mise_" + x].sum()
#
# print(total)










