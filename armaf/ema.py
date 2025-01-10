#!/usr/bin/env python

import os
from pprint import pformat

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .trader import APIResponseError


def optimize_ema_span(
    api,
    account_id,
    instruments,
    min_span=5,
    max_span=60,
    n_splits=10,
    granularity="M1",
    count=5000,
    complete=True,
):
    spans = np.arange(min_span, max_span + 1, 1)
    df_span = pd.DataFrame(
        [
            {
                "instrument": i,
                **_select_best_ema_span(
                    series=_fetch_log_return(
                        api=api,
                        instrument=i,
                        granularity=granularity,
                        count=count,
                        complete=complete,
                    ),
                    spans=spans,
                    n_splits=n_splits,
                ),
            }
            for i in instruments
        ]
    ).set_index("instrument", drop=True)
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df_span)


def _select_best_ema_span(series, spans, n_splits=10):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    split_indices = list(tscv.split(series))
    train_index, test_index = split_indices[
        np.array(
            [
                _calculate_sharpe_ratio(
                    series=series.iloc[
                        np.concatenate((train_index, test_index), axis=None)
                    ]
                )
                for train_index, test_index in split_indices
            ]
        ).argmax()
    ]
    mses = pd.Series(
        {
            s: _calculate_ema_mse(
                train_series=series.iloc[train_index],
                test_series=series.iloc[test_index],
                span=s,
            )
            for s in spans
        }
    )
    span = mses.idxmin()
    return {
        "span": span,
        "mse": mses[span],
        "train_size": train_index.size,
        "test_size": test_index.size,
    }


def _calculate_sharpe_ratio(series):
    return abs((series.iloc[-1] - series.iloc[0]) / series.std())


def _calculate_ema_mse(train_series, test_series, span):
    ema_train = train_series.ewm(span=span, adjust=False, ignore_na=True).mean()
    return (test_series - ema_train.iloc[-1]).pow(2).mean()


def _fetch_log_return(api, instrument, granularity="S5", count=5000, complete=True):
    res = api.instrument.candles(
        instrument=instrument,
        price="BA",
        granularity=granularity,
        count=min(5000, int(count)),
    )
    if "candles" in res.body:
        return (
            pd.DataFrame(
                [
                    {
                        "time": c.time,
                        "bid": c.bid.c,
                        "ask": c.ask.c,
                        "volume": c.volume,
                        "complete": c.complete,
                    }
                    for c in res.body["candles"]
                    if c.complete or not complete
                ]
            )
            .assign(
                log_return=lambda d: np.log(d[["ask", "bid"]].mean(axis=1)).diff(),
                time=lambda d: pd.to_datetime(d["time"]),
                instrument=instrument,
            )
            .set_index("time", drop=True)["log_return"]
            .dropna()
        )
    else:
        raise APIResponseError("unexpected response:" + os.linesep + pformat(res.body))
