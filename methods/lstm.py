import matplotlib.pyplot as plt
from math import floor, ceil, sqrt
import numpy as np
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 10
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from datetime import date, datetime, timedelta


def lstm_prediction(df, ticker_name, save_id, interval, epoch=1, batch_size=1, need_future=False):
    shape = df.shape[0]
    df_new = df[['Close']]
    df_new.head()
    dataset = df_new.values
    train = df_new[:ceil(shape * 0.75)]

    valid = df_new[ceil(shape * 0.75):]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    x_train, y_train = [], []
    for i in range(40, len(train)):
        x_train.append(scaled_data[i - 40:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    try:
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    except Exception:
        return {'error': 'not enough data for predict'}

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, verbose=0)
    inputs = df_new[len(df_new) - len(valid) - 40:].values
    # print(df_new[len(df_new) - len(valid) - 40:])

    inputs = inputs.reshape(-1, 1)
    # print(inputs)
    inputs = scaler.transform(inputs)
    X_test = []

    for i in range(40, inputs.shape[0]):
        X_test.append(inputs[i - 40:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)
    rms = np.sqrt(np.mean(np.power((valid - closing_price), 2)))
    valid['Predictions'] = closing_price

    if need_future:
        d = 4
        x_input = inputs[len(inputs)//d:].reshape(1, -1)
        temp_input = x_input[0].tolist()

        lst_output = []
        n_steps = len(inputs) - len(inputs)//d
        i = 0
        while i < 30:
            if len(temp_input) > n_steps:
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                lst_output.extend(yhat.tolist())
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i = i + 1

        start_time = valid.index[-1]
        # 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        match interval:
            case '1m':
                delta = timedelta(minutes=1)
            case '2m':
                delta = timedelta(minutes=2)
            case '5m':
                delta = timedelta(minutes=5)
            case '15m':
                delta = timedelta(minutes=15)
            case '30m':
                delta = timedelta(minutes=30)
            case '1h':
                delta = timedelta(hours=1)
            case '1d':
                delta = timedelta(days=1)
            case '5d':
                delta = timedelta(days=5)
            case '1wk':
                delta = timedelta(weeks=1)
            case _:
                delta = timedelta(hours=1)

        # delta = timedelta(hours=1)
        new_axis = []
        for i in range(len(lst_output)):
            new_axis.append(start_time + delta * i)

    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])

    if need_future:
        plt.plot(new_axis, scaler.inverse_transform(lst_output), color='red')

    plt.xlabel('Date', size=20)
    plt.ylabel('Stock Price', size=20)
    plt.title(f'Stock Price Prediction by Long Short Term Memory (LSTM) ({ticker_name})', size=20)
    plt.legend(['Model Training Data', 'Actual Data', 'Predicted Data', 'Predicted Future Data'])
    plt.savefig(f'charts/{save_id}.png')
    plt.close()
    return {'shape_training_set': train.shape[0], 'shape_validation_set': valid.shape[0], 'rms': rms}


def datetime_range(start, end, delta):
    current = start
    if not isinstance(delta, timedelta):
        delta = timedelta(**delta)
    while current < end:
        yield current
        current += delta
