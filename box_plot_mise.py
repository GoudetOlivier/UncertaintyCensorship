import plotly.graph_objs as go
import plotly.offline as offline
# offline.init_notebook_mode(connected = True)
import pandas as pd

# list_model = ["True_proba", "Standard_beran", "Subramanian","Linear_MAR_with_delta", "NN_MAR_with_delta", "Linear_with_delta", "NN_two_steps_with_delta"]
# list_name = [ "Oracle", "Standard", "MAR - Subramanian type", "MAR - Linear model", "MAR - Neural Network", "MNAR - Linear model", "MNAR - Neural Network"]
# list_color = ['royalblue', 'orange', 'pink', 'brown', 'red', 'yellow','green']

list_model = ["True_proba", "Standard_beran", "Subramanian", "NN_MAR_with_delta",  "NN_two_steps_with_delta"]
list_name = [ "Oracle", "Standard", "MAR - Subramanian type",  "MAR - Neural Network",  "MNAR - Neural Network"]
list_color = ['royalblue', 'orange', 'pink',  'red', 'green']

list_model_linear = ["Linear_MAR_with_delta", "Linear_with_delta"]


dico = {}
for model in list_model:
    dico[model] = []


x = []

for rho in ["0","0.25","0.5", "0.75"]:
    x.extend(r'$\large{\rho=' + rho + '}$' for i in range(200))


    dataPath = "save_results/weibull/Global_score_weibull_n_1000_nbIter_200_rho_" + rho + "_noCovariateMode_False.csv"
    df = pd.read_csv(dataPath)

    dataPath = "save_results/weibull/linear/Global_score_Linear_weibull_n_1000_nbIter_20_rho_" + rho + "_noCovariateMode_False.csv"
    df_linear = pd.read_csv(dataPath)

    for model in list_model:

        if(model in list_model_linear):
            for i in range(10):
                dico[model].extend(list(df_linear[model].values * 300))
        else:
            dico[model].extend(list(df[model].values*300))


data = []
for idx, model in enumerate(list_model):

    data.append(go.Box(
        y=dico[model],
        x=x,
        name=list_name[idx],
        marker=dict(
            color=list_color[idx]
        )
    ))


# trace1 = go.Box(
#     y=[0.6, 0.7, 0.3, 0.6, 0.0, 0.5, 0.7, 0.9, 0.5, 0.8, 0.7, 0.2],
#     x=x,
#     name='radishes',
#     marker=dict(
#         color='#FF4136'
#     )
# )
# trace2 = go.Box(
#     y=[0.6, 0.7, 0.3, 0.6, 0.0, 0.5, 0.7, 0.9, 0.5, 0.8, 0.7, 0.2],
#     x=x,
#     name='carrots',
#     marker=dict(
#         color='#FF851B'
#     )
# )

# data = [trace0, trace1, trace2]
layout = go.Layout(
    yaxis=go.layout.YAxis(
        title='MISE',
        range=[0,3]
        #zeroline=False
    ),
    font=dict(
        family="Courier New, monospace",
        size=24
    ),
    boxmode='group'
)

fig = go.Figure(data=data, layout=layout)

fig.show()
