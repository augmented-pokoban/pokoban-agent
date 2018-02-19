from autoencoder.EncoderData import load_object
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np

data = load_object('agent_error_data.pkl.zip')

means = np.mean(data, axis=1)

x_axis = [x for x in range(1,6)]


trace_agent_error = go.Scatter(
    x=x_axis,
    y=means,
    name='Avg. agent error'
)

trace = [trace_agent_error]

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
        tickfont=dict(
            size=16,
            color='black'
        ),
    ),
    yaxis=dict(
        title='Mean agent errors',
        titlefont=dict(
            size=18,
            color='darkgrey'
        ),
        dtick=0.5,
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

fig = go.Figure(data=trace, layout=layout)

py.plot(fig, image='png')
# py.plot(fig)
