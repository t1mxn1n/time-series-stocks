import matplotlib.pyplot as plt
from math import floor, ceil, sqrt
import numpy as np
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 10


def moving_avg_prediction(df, ticker_name, save_id, interval=None):
    shape = df.shape[0]
    df_new = df[['Close']]
    df_new.head()
    train_set = df_new.iloc[:ceil(shape * 0.75)]
    valid_set = df_new.iloc[ceil(shape * 0.75):]
    # print('Shape of Training Set', train_set.shape)
    # print('Shape of Validation Set', valid_set.shape)
    preds = []
    for i in range(0, valid_set.shape[0]):
        a = train_set['Close'][len(train_set) - valid_set.shape[0] + i:].sum() + sum(preds)
        b = a / (valid_set.shape[0])
        preds.append(b)
    rms = np.sqrt(np.mean(np.power((np.array(valid_set['Close']) - preds), 2)))
    valid_set['Predictions'] = preds
    plt.plot(train_set['Close'])
    plt.plot(valid_set[['Close', 'Predictions']])
    plt.xlabel('Date', size=20)
    plt.ylabel('Stock Price', size=20)
    plt.title(f'Stock Price Prediction by Moving Averages ({ticker_name})', size=20)
    plt.legend(['Model Training Data', 'Actual Data', 'Predicted Data'])
    plt.savefig(f'charts/{save_id}.png')
    plt.close()
    return {'shape_training_set': train_set.shape[0], 'shape_validation_set': valid_set.shape[0], 'rms': rms}
