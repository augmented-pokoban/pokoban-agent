import plotly.offline as py
import plotly.graph_objs as go
import pandas as pd

test = pd.read_csv('encoder_test.csv')
train = pd.read_csv('encoder_train.csv')

print(test)

x_test = test['Step'].values.tolist()
y_test = test['Value'].values.tolist()
x_train = train['Step'].values.tolist()
y_train = train['Value'].values.tolist()


trace_test = go.Scatter(
    x=x_test,
    y=y_test,
    name='Test'
)

trace_train = go.Scatter(
    x=x_train,
    y=y_train,
    name='Train',

)

data = [trace_test, trace_train]

layout = go.Layout(
    autosize=False,
    width=800,
    height=500,
    xaxis=dict(
        title='Episode',
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
        title='Mean squared error',
        titlefont=dict(
            size=18,
            color='darkgrey'
        ),
        dtick=0.05,
        range=[0.0,0.4],
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
# py.plot(fig)
