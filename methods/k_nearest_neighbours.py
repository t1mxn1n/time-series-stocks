import matplotlib.pyplot as plt
from math import floor, ceil, sqrt
import numpy as np
import datetime as dt
import pandas as pd
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 10
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV


def k_nearest_neighbours_predict(df, ticker_name, save_id, interval=None):

    scaler = MinMaxScaler(feature_range=(0, 1))
    shape = df.shape[0]
    df_new = df[['Close']]
    train_set = df_new.iloc[:ceil(shape * 0.75)]
    valid_set = df_new.iloc[ceil(shape * 0.75):]
    # print('Shape of Training Set', train_set.shape)
    # print('Shape of Validation Set', valid_set.shape)
    train = train_set.reset_index()
    valid = valid_set.reset_index()

    field_time = 'Datetime'
    try:
        x_train = train[field_time].map(dt.datetime.toordinal)
    except KeyError:
        field_time = 'Date'
        x_train = train[field_time].map(dt.datetime.toordinal)

    y_train = train[['Close']]
    x_valid = valid[field_time].map(dt.datetime.toordinal)
    y_valid = valid[['Close']]
    x_train_scaled = scaler.fit_transform(np.array(x_train).reshape(-1, 1))
    x_train = pd.DataFrame(x_train_scaled)
    x_valid_scaled = scaler.fit_transform(np.array(x_valid).reshape(-1, 1))
    x_valid = pd.DataFrame(x_valid_scaled)
    params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9]}
    knn = neighbors.KNeighborsRegressor()
    model = GridSearchCV(knn, params, cv=5)
    model.fit(x_train, y_train)
    preds = model.predict(x_valid)
    rms = np.sqrt(np.mean(np.power((np.array(y_valid) - np.array(preds)), 2)))
    valid_set['Predictions'] = preds
    plt.plot(train_set['Close'])
    plt.plot(valid_set[['Close', 'Predictions']])
    plt.xlabel('Date', size=20)
    plt.ylabel('Stock Price', size=20)
    plt.title(f'Stock Price Prediction by K-Nearest Neighbors ({ticker_name})', size=20)
    plt.legend(['Model Training Data', 'Actual Data', 'Predicted Data'])
    plt.savefig(f'charts/{save_id}.png')
    plt.close()
    return {'shape_training_set': train_set.shape[0], 'shape_validation_set': valid_set.shape[0], 'rms': rms}
