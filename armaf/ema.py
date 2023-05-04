#!/usr/bin/env python

import os
from pprint import pformat

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from .trader import APIResponseError


def optimize_ema_span(api, account_id, instruments, min_span=5, max_span=60,
                      n_splits=10, granularity='M1', count=5000,
                      complete=True):
    spans = np.arange(min_span, max_span + 1, 1)
    df_span = pd.DataFrame([
        {
            'instrument': i,
            **_select_best_ema_spans(
                series=_fetch_candle_df(
                    api=api, instrument=i, granularity=granularity,
                    count=count, complete=complete
                )['mid'],
                spans=spans, n_splits=n_splits
            )
        } for i in instruments
    ]).set_index('instrument', drop=True).drop(columns='count')
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None):
        print(df_span)


def _select_best_ema_spans(series, spans, n_splits=10):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return pd.Series([
        pd.Series({
            s: _calculate_ema_mse(
                train_series=series.iloc[train_index],
                test_series=series.iloc[test_index], span=s
            ) for s in spans
        }).idxmax()
        for train_index, test_index in tscv.split(series)
    ]).describe().to_dict()


def _calculate_ema_mse(train_series, test_series, span):
    ema_train = train_series.ewm(
        span=span, adjust=False, ignore_na=True
    ).mean()
    ema_test = ema_train.iloc[-1]
    ema_test = pd.concat([ema_train, test_series]).ewm(
        span=span, adjust=False, ignore_na=True
    ).mean().iloc[len(ema_train):]
    return mean_squared_error(test_series, ema_test)


def _fetch_candle_df(api, instrument, granularity='S5', count=5000,
                     complete=True):
    res = api.instrument.candles(
        instrument=instrument, price='BA', granularity=granularity,
        count=min(5000, int(count))
    )
    if 'candles' in res.body:
        return pd.DataFrame([
            {
                'time': c.time, 'bid': c.bid.c, 'ask': c.ask.c,
                'volume': c.volume, 'complete': c.complete
            } for c in res.body['candles'] if c.complete or not complete
        ]).assign(
            mid=lambda d: d[['ask', 'bid']].mean(axis=1),
            time=lambda d: pd.to_datetime(d['time']), instrument=instrument
        ).set_index('time', drop=True)
    else:
        raise APIResponseError(
            'unexpected response:' + os.linesep + pformat(res.body)
        )
