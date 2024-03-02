#!/usr/bin/env python

import logging
import os
from pprint import pformat

import numpy as np

from .arima import OptimizedArima


class SignalDetector(object):
    def __init__(self, trigger_ssr_window=900, trigger_test_size=900,
                 trigger_forecast_size=60, trend_ssr_window=900,
                 trend_test_size=900, trend_forecast_size=60,
                 significance_level=0.01, arima_p_range=(0, 1),
                 arima_d_range=(0, 1), arima_q_range=(0, 1), volume_factor=0):
        self.__logger = logging.getLogger(__name__)
        self.trigger_ssr_window = trigger_ssr_window
        self.trigger_test_size = trigger_test_size
        self.trigger_forecast_size = trigger_forecast_size
        self.trend_ssr_window = trend_ssr_window
        self.trend_test_size = trend_test_size
        self.trend_forecast_size = trend_forecast_size
        self.significance_level = significance_level
        self.arima_p_range = arima_p_range
        self.arima_d_range = arima_d_range
        self.arima_q_range = arima_q_range
        for k in 'pdq':
            a, b = getattr(self, f'arima_{k}_range')
            if not (0 <= a <= b):
                raise ValueError(f'invalid ARIMA {k} range')
        self.volume_factor = volume_factor
        self.min_sample_size = (
            max(self.trigger_ssr_window, self.trend_ssr_window)
            + max(self.trigger_test_size, self.trigger_test_size)
        )
        self.trigger_ssr_arima_order = (None, None, None)
        self.trend_ssr_arima_order = (None, None, None)
        self.__logger.debug('vars(self):' + os.linesep + pformat(vars(self)))

    def detect(self, df_history, position_side=None):
        df_lrr = self._calculate_log_return_rate(
            df_history=df_history, volume_factor=self.volume_factor
        ).reset_index()
        self.__logger.info(f'df_lrr.shape: {df_lrr.shape}')
        self.__logger.debug(f'df_lrr:{os.linesep}{df_lrr}')
        if df_lrr.shape[0] < self.min_sample_size:
            trigger_ssr = trend_ssr = np.nan
            trigger_ssr_ci = trend_ssr_ci = np.array([np.nan, np.nan])
        else:
            (
                trigger_ssr, trigger_ssr_ci, self.trigger_ssr_arima_order
            ) = self._calculate_forecast_ci(
                y=self._calculate_signed_sharpe_ratio(
                    df_lrr=df_lrr, window=self.trigger_ssr_window
                ).tail(self.trigger_ssr_window + self.trigger_test_size),
                test_size=self.trigger_test_size,
                forecast_size=self.trigger_forecast_size,
                significance_level=self.significance_level,
                arima_p_range=self.arima_p_range,
                arima_d_range=self.arima_d_range,
                arima_q_range=self.arima_q_range
            )
            (
                trend_ssr, trend_ssr_ci, self.trend_ssr_arima_order
            ) = self._calculate_forecast_ci(
                y=self._calculate_signed_sharpe_ratio(
                    df_lrr=df_lrr, window=self.trend_ssr_window
                ).tail(self.trend_ssr_window + self.trend_test_size),
                test_size=self.trend_test_size,
                forecast_size=self.trend_forecast_size,
                significance_level=self.significance_level,
                arima_p_range=self.arima_p_range,
                arima_d_range=self.arima_d_range,
                arima_q_range=self.arima_q_range
            )
        self.__logger.info(
            'trend_ssr, trend_ssr_ci, trigger_ssr, trigger_ssr_ci:'
            + f' {trend_ssr}, {trend_ssr_ci}, {trigger_ssr}, {trigger_ssr_ci}'
        )
        if trigger_ssr_ci[0] > 0 and trend_ssr > 0:
            act = 'long'
        elif trigger_ssr_ci[1] < 0 and trend_ssr < 0:
            act = 'short'
        elif ((position_side == 'short'
               and ((trigger_ssr > 0 and trend_ssr > 0)
                    or trigger_ssr_ci[0] > 0 or trend_ssr_ci[0] > 0))
              or (position_side == 'long'
                  and ((trigger_ssr < 0 and trend_ssr < 0)
                       or trigger_ssr_ci[1] < 0 or trend_ssr_ci[1] < 0))):
            act = 'closing'
        else:
            act = None
        self.__logger.info(f'act: {act}')
        return {
            'trend_ssr': trend_ssr, 'trend_ssr_ci_lower': trend_ssr_ci[0],
            'trend_ssr_ci_upper': trend_ssr_ci[1], 'trigger_ssr': trigger_ssr,
            'trigger_ssr_ci_lower': trigger_ssr_ci[0],
            'trigger_ssr_ci_upper': trigger_ssr_ci[1], 'act': act,
            'log_str': '{0:^52}|{1:^52}|'.format(
                'SSR({0},{1},{2};{3}):{4:>9}{5:>16}'.format(
                    self.trigger_ssr_window, self.trigger_test_size,
                    self.trigger_forecast_size,
                    ','.join(
                        str(
                            getattr(self, f'arima_{k}_range')[0]
                            if v is None else v
                        ) for k, v in zip('pdq', self.trigger_ssr_arima_order)
                    ),
                    '{:.1g}'.format(trigger_ssr),
                    np.array2string(
                        trigger_ssr_ci,
                        formatter={'float_kind': lambda f: f'{f:.1g}'}
                    )
                ),
                'SSR({0},{1},{2};{3}):{4:>9}{5:>16}'.format(
                    self.trend_ssr_window, self.trend_test_size,
                    self.trend_forecast_size,
                    ','.join(
                        str(
                            getattr(self, f'arima_{k}_range')[0]
                            if v is None else v
                        ) for k, v in zip('pdq', self.trend_ssr_arima_order)
                    ),
                    '{:.1g}'.format(trend_ssr),
                    np.array2string(
                        trend_ssr_ci,
                        formatter={'float_kind': lambda f: f'{f:.1g}'}
                    )
                )
            )
        }

    @staticmethod
    def _calculate_log_return_rate(df_history, volume_factor=0):
        return df_history.assign(
            log_return=lambda d: np.log(d[['ask', 'bid']].mean(axis=1)).diff(),
            delta_sec=lambda d: d.index.to_series().diff().dt.total_seconds(),
            volume_weight=lambda d:
            np.power(d['volume'], volume_factor).pipe(lambda s: (s / s.mean()))
        ).assign(
            log_return_rate=lambda d:
            (d['log_return'] / d['delta_sec'] * d['volume_weight']).fillna(0)
        )

    @staticmethod
    def _calculate_signed_sharpe_ratio(df_lrr, window):
        pl_ratio_ewm = (
            np.exp(df_lrr['log_return_rate']) - 1
        ).ewm(span=window, adjust=False, ignore_na=True)
        return (
            pl_ratio_ewm.mean() * df_lrr['bid'] / df_lrr['ask']
            / pl_ratio_ewm.std()
        )

    @staticmethod
    def _calculate_forecast_ci(y, test_size, forecast_size,
                               significance_level=0.01, arima_p_range=(0, 1),
                               arima_d_range=(0, 1), arima_q_range=(0, 1)):
        oa = OptimizedArima(
            y=y, test_size=test_size, p_range=arima_p_range,
            d_range=arima_d_range, q_range=arima_q_range,
            model_kw={
                'enforce_stationarity': False, 'enforce_invertibility': False
            }
        )
        forecasts = oa.forecast_frame(
            alpha=significance_level, get_forecast_kw={'steps': forecast_size}
        ).iloc[-1]
        return (
            forecasts['mean'],
            np.array([forecasts['mean_ci_lower'], forecasts['mean_ci_upper']]),
            oa.arima_order
        )
