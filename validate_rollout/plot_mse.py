import plotly.offline as py
import plotly.graph_objs as go
import numpy as np

from autoencoder.EncoderData import load_object

# data = load_object('mse_data.pkl.zip')
mse_data = load_object('mse_data.pkl.zip')

mean_data = np.mean(mse_data, axis=1)


x_axis = []
y_axis = [y for x in mse_data for y in x]

for i in range(5):
    # create range
    x_axis += [i + 1] * len(mse_data[i])

trace_mse = go.Scatter(
    x=x_axis,
    y=y_axis,
    mode='markers',
    name='MSE'
)

mean_x = [x for x in range(1, 6)]

print(mean_x)

trace_mean = go.Scatter(
    x=mean_x,
    y=mean_data,
    name='MSE average'
)

data = [trace_mse, trace_mean]

layout = go.Layout(
    autosize=False,
    width=800,
    height=500,
    xaxis=dict(
        title='Rollout step',
        titlefont=dict(
            size=18,
            color='darkgrey'
        ),
        autotick=False,
        ticks='outside',
        tick0=0,
        dtick=1,
        tickfont=dict(
            size=16,
            color='black'
        ),
    ),
    yaxis=dict(
        title='Mean squared error',
        titlefont=dict(
            size=18,
            color='darkgrey'
        ),
        range=[0.0,0.4],
        dtick=0.1,
        tickfont=dict(
            size=16,
            color='black'
        ),
    ),
    legend=dict(
        font=dict(
            size=16
        )
    )
)

fig = go.Figure(data=data, layout=layout)

py.plot(fig, image='png')
