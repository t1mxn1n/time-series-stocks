from math import floor, ceil

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 10
import numpy as np
import pandas as pd

from pmdarima.arima import auto_arima


def auto_arima_prediction(df, ticker_name, save_id, interval=None):
    shape = df.shape[0]
    df_new = df
    data = df_new.sort_index(ascending=True, axis=0)
    train_set = data[:ceil(shape * 0.75)]
    valid_set = data[ceil(shape * 0.75):]
    # print('Shape of Training Set', train_set.shape)
    # print('Shape of Validation Set', valid_set.shape)
    training = train_set['Close']
    model = auto_arima(training, start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, seasonal=True, d=1, D=1,
                       trace=True, error_action='ignore', suppress_warnings=True)
    model.fit(training)
    forecast = model.predict(n_periods=ceil(floor(df.shape[0] * 0.25)))
    forecast.index = valid_set.index
    forecast = pd.DataFrame(forecast, columns=['Prediction'])
    rms = np.sqrt(np.mean(np.power((np.array(valid_set['Close']) - np.array(forecast['Prediction'])), 2)))
    plt.plot(train_set['Close'])
    plt.plot(valid_set['Close'])
    plt.plot(forecast['Prediction'])
    plt.xlabel('Date', size=20)
    plt.ylabel('Stock Price', size=20)
    plt.title(f'Stock Price Prediction by Auto ARIMA ({ticker_name})', size=20)
    plt.legend(['Model Training Data', 'Actual Data', 'Predicted Data'])
    plt.savefig(f'charts/{save_id}.png')
    plt.close()
    return {'shape_training_set': train_set.shape[0], 'shape_validation_set': valid_set.shape[0], 'rms': rms}
