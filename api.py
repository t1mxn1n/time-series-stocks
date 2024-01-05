import random
import os
import uvicorn
from fastapi import FastAPI, Query, Response
from datetime import datetime
import yfinance as yf

from methods import (
    arima, k_nearest_neighbours, linear_regression, lstm, moving_average, prophet_method
)
from imgur_api import imgur_upload

import warnings
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore")

app = FastAPI(title='Тимонин Никита 8ВМ32 Курсовой проект')
methods_import = {
    0: moving_average.moving_avg_prediction,
    1: arima.auto_arima_prediction,
    2: k_nearest_neighbours.k_nearest_neighbours_predict,
    3: linear_regression.linear_regression_prediction,
    4: lstm.lstm_prediction,
    5: prophet_method.fb_prophet_prediction
}


@app.get("/predict", description='Предсказывание цен акций')
async def predict(
        response: Response,
        stock: str = Query('MSFT', description='Название акции, например TSLA, GOOGL, AAPL, NVDA, AMZN'),
        period: str = Query(...,
                            enum=['1d', '1h', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'],
                            description='Временные границы графика'),
        interval: str = Query(...,
                              enum=['1m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'],
                              description='Группировка котировок'),
        method: int = Query(..., enum=[0, 1, 2, 3, 4, 5],
                            description=('0 - Moving Average; 1 - ARIMA; 2 - K-Nearest Neighbours; '
                                         '3 - Linear Regression; 4 - LSTM; 5 - Prophet'))
) -> dict:
    response.headers["Access-Control-Allow-Origin"] = os.getenv('cors')
    ticker = yf.Ticker(stock)
    history = ticker.history(period=period, interval=interval)
    if history.empty:
        return {'incorrect params': f'{stock}, period: {period}, interval: {interval}',
                'error_msg': 'empty dataframe'}

    if methods_import.get(method):
        use_method = methods_import[method]
    else:
        return {'error': f'no such method {method}'}
    start_time = datetime.now()
    random_id = random.randint(100000, 999999)
    try:
        process_info = use_method(history, stock, random_id, interval)
    except Exception as e:
        return {'processing_error': e,
                'params': f'{stock}, {period}, {interval}'}
    time_processing = datetime.now() - start_time

    if process_info.get('error'):
        return process_info

    url = imgur_upload(f'charts/{random_id}.png')

    data = {'url': url,
            'time_processing': str(time_processing),
            'details': process_info}
    return data


@app.get("/lstm", description='Долгая краткосрочная память (Long short-term memory - LSTM')
async def lstm_endpoint(
        response: Response,
        stock: str = Query('MSFT', description='Название акции, например TSLA, GOOGL, AAPL, NVDA, AMZN'),
        period: str = Query(...,
                            enum=['1d', '1h', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'],
                            description='Временные границы графика'),
        interval: str = Query(...,
                              enum=['1m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'],
                              description='Группировка котировок'),
        need_future_predict: bool = Query(False, enum=[False, True]),
        epoch: int = Query(1, description='Количество эпох сети'),
        batch_size: int = Query(1, description='Размер батча')
) -> dict:
    response.headers["Access-Control-Allow-Origin"] = os.getenv('cors')
    ticker = yf.Ticker(stock)
    history = ticker.history(period=period, interval=interval)
    if history.empty:
        return {'incorrect params': f'{stock}, period: {period}, interval: {interval}',
                'error_msg': 'empty dataframe'}

    start_time = datetime.now()
    random_id = random.randint(100000, 999999)
    try:
        # print(stock, random_id, epoch, batch_size, need_future_predict)
        process_info = lstm.lstm_prediction(history, stock, random_id, interval, epoch, batch_size, need_future_predict)
    except Exception as e:
        return {'processing_error': e,
                'params': f'{stock}, {period}, {interval}'}
    time_processing = datetime.now() - start_time

    if process_info.get('error'):
        return process_info

    url = imgur_upload(f'charts/{random_id}.png')

    data = {'url': url,
            'time_processing': str(time_processing),
            'details': process_info}
    return data


if __name__ == "__main__":
    uvicorn.run('api:app', host='127.0.0.1', port=8000, reload=True)
