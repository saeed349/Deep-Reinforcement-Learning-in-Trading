import plotly.offline as py
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *
import pandas as pd
import numpy as np
py.init_notebook_mode()

def loss_vis(df,mean=True,N=10):
    if(mean):
        x = df.index[N-1:]
        y1 = np.convolve(df.loss, np.ones((N,))/N, mode='valid')
        y2 = np.convolve(df.val_loss, np.ones((N,))/N, mode='valid')
    else:
        x = df.index
        y1 = df.loss
        y2 = df.val_loss
        
    trace0 = Scatter(
    x = x,
    y = y1,
    name = 'Training',
    line = dict(
        color = ('rgb(205, 12, 24)'),
        width = 4)
    )
    trace1 = Scatter(
        x = x,
        y = y2,
        name = 'Validation',
        line = dict(
            color = ('rgb(22, 96, 167)'),
            width = 4,)
    )


    data = [trace0, trace1]

    # Edit the layout
    layout = dict(title = 'Training vs Validation Loss',
                  xaxis = dict(title = 'Epoch'),
                  yaxis = dict(title = 'Loss'),
                  )

    fig = dict(data=data, layout=layout)
    py.iplot(fig, filename='styled-line')
    
def reward_vis(df, mean=True, N=10):
    if(mean):
        x = df.index[N-1:]
        y1 = np.convolve(df.reward, np.ones((N,))/N, mode='valid')
    else:
        x = df.index
        y1 = df.reward
        
    trace0 = Scatter(
    x = df.index[N-1:],
    y = np.convolve(df.reward, np.ones((N,))/N, mode='valid'),
    name = 'Reward',
    line = dict(
        color = ('rgb(205, 12, 24)'),
        width = 4)
    )

    data = [trace0]

    # Edit the layout
    layout = dict(title = 'Reward',
                  xaxis = dict(title = 'Epoch'),
                  yaxis = dict(title = 'Reward'),
                  )

    fig = dict(data=data, layout=layout)
    py.iplot(fig, filename='styled-line')