import matplotlib.pyplot as plt
from math import ceil
import numpy as np
from prophet import Prophet
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 10


def fb_prophet_prediction(df, ticker_name, save_id, interval=None):

    shape = df.shape[0]
    df_new = df[['Close']]
    df_new.reset_index(inplace=True)
    field_time = 'Datetime'
    try:
        df_new[field_time] = df_new['Datetime'].dt.tz_localize(None)
    except KeyError:
        field_time = 'Date'
        df_new[field_time] = df_new['Date'].dt.tz_localize(None)

    df_new.rename(columns={'Close': 'y', field_time: 'ds'}, inplace=True)
    train_set = df_new.iloc[:ceil(shape * 0.75)]
    valid_set = df_new.iloc[ceil(shape * 0.75):]
    # print('Shape of Training Set', train_set.shape)
    # print('Shape of Validation Set', valid_set.shape)
    model = Prophet()
    model.fit(train_set)
    close_prices = model.make_future_dataframe(periods=len(valid_set))
    forecast = model.predict(close_prices)
    forecast_valid = forecast['yhat'][ceil(shape * 0.75):]
    rms = np.sqrt(np.mean(np.power((np.array(valid_set['y']) - np.array(forecast_valid)), 2)))
    valid_set['Predictions'] = forecast_valid.values
    plt.plot(train_set['y'])
    plt.plot(valid_set[['y', 'Predictions']])
    plt.xlabel('Date', size=20)
    plt.ylabel('Stock Price', size=20)
    plt.title(f'Stock Price Prediction by Prophet ({ticker_name})', size=20)
    plt.legend(['Model Training Data', 'Actual Data', 'Predicted Data'])
    plt.savefig(f'charts/{save_id}.png')
    plt.close()
    return {'shape_training_set': train_set.shape[0], 'shape_validation_set': valid_set.shape[0], 'rms': rms}
