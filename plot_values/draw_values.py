import plotly.offline as py
import plotly.graph_objs as go

from autoencoder.EncoderData import load_object

values = load_object('values_20_steps.pkl.zip')
# values = load_object('adv_r_v.pkl.zip')

x_axis = [x for x in range(1,21)]

print(values)

trace_values = go.Scatter(
    x=x_axis,
    y=values,
    name='Values'
)
#
# trace_value_diff = go.Scatter(
#     x=x_axis,
#     y=values['value_diff'],
#     name='Value diff'
# )
#
# trace_adv_disc = go.Scatter(
#     x=x_axis,
#     y=values['adv_disc'],
#     name='Advantages disc'
# )
#
# trace_rewards = go.Scatter(
#     x=x_axis,
#     y=values['rewards'],
#     name='Rewards disc'
# )

trace = [trace_values]

layout = go.Layout(
    autosize=False,
    width=800,
    height=500,
    xaxis=dict(
        title='Step',
        dtick=2,
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
        title='V(s)',
        titlefont=dict(
            size=18,
            color='darkgrey'
        ),
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

fig = go.Figure(data=trace, layout=layout)

py.plot(fig, image='png')
# py.plot(fig)