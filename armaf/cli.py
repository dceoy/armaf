#!/usr/bin/env python
"""
ARIMA-based Forex Trader using Oanda API

Usage:
    armaf -h|--help
    armaf --version
    armaf close [--debug|--info] [--oanda-account=<id>] [--oanda-token=<str>]
        [--oanda-env=<str>] [<instrument>...]
    armaf trade [--debug|--info] [--oanda-account=<id>] [--oanda-token=<str>]
        [--oanda-env=<str>] [--quiet] [--granularity=<str>]
        [--betting-strategy=<str>] [--margin-lack-reset]
        [--scanned-transaction-count=<int>] [--unit-margin=<ratio>]
        [--preserved-margin=<ratio>] [--take-profit-limit=<float>]
        [--trailing-stop-limit=<float>] [--stop-loss-limit=<float>]
        [--max-spread=<float>] [--sleeping=<ratio>]
        [--trigger-ssr-window=<int>] [--trigger-test-size=<int>]
        [--trigger-forecast-size=<int>] [--trend-ssr-window=<int>]
        [--trend-test-size=<int>] [--trend-forecast-size=<int>]
        [--significance-level=<float>] [--min-arima-p=<int>]
        [--max-arima-p=<int>] [--min-arima-d=<int>] [--max-arima-d=<int>]
        [--min-arima-q=<int>] [--max-arima-q=<int>] [--volume-factor=<float>]
        [--retry-count=<int>] [--dry-run] <instrument>...
    armaf optimize [--debug|--info] [--oanda-account=<id>]
        [--oanda-token=<str>] [--oanda-env=<str>] [--quiet]
        [--min-span=<int>] [--max-span=<int>] [--n-splits=<int>]
        [--granularity=<str>] [--count=<int>] <instrument>...

Options:
    -h, --help          Print help and exit
    --version           Print version and exit
    --debug, --info     Execute a command with debug|info messages
    --oanda-account=<id>
                        Set an Oanda account ID [$OANDA_ACCOUNT]
    --oanda-token=<str>
                        Set an Oanda API token [$OANDA_TOKEN]
    --oanda-env=<str>   Set an Oanda trading environment [default: trade]
                        { trade, practice }
    --quiet             Suppress messages
    --granularity=<str>
                        Set the granularities [default: S5]
    --betting-strategy=<str>
                        Set the betting strategy [default: constant]
                        { constant, martingale, paroli, dalembert,
                          oscarsgrind }
    --margin-lack-reset
                        Reset betting on margin lack
    --scanned-transaction-count=<int>
                        Set the transaction count to scan [default: 0]
    --unit-margin=<ratio>
                        Set the unit margin ratio to NAV [default: 0.01]
    --preserved-margin=<ratio>
                        Set the preserved margin ratio [default: 0.01]
    --take-profit-limit=<float>
                        Set the take-profit limit ratio [default: 0.01]
    --trailing-stop-limit=<float>
                        Set the trailing-stop limit ratio [default: 0.01]
    --stop-loss-limit=<float>
                        Set the stop-loss limit ratio [default: 0.01]
    --max-spread=<float>
                        Set the max spread ratio [default: 0.01]
    --sleeping=<ratio>  Set the daily sleeping ratio [default: 0]
    --trigger-ssr-window=<int>
                        Set the window for trigger SSR signal [default: 12]
    --trigger-test-size=<int>
                        Set the trigger test step size [default: 180]
    --trigger-forecast-size=<int>
                        Set the trigger forecast step size [default: 12]
    --trend-ssr-window=<int>
                        Set the window for trend SSR signal [default: 60]
    --trend-test-size=<int>
                        Set the trend test step size [default: 180]
    --trend-forecast-size=<int>
                        Set the trend forecast step size [default: 12]
    --significance-level=<float>
                        Set the significance level [default: 0.01]
    --min-arima-p=<int>
                        Set the min AR component of ARIMA [default: 0]
    --max-arima-p=<int>
                        Set the max AR component of ARIMA [default: 0]
    --min-arima-d=<int>
                        Set the min I component of ARIMA [default: 0]
    --max-arima-d=<int>
                        Set the max I component of ARIMA [default: 0]
    --min-arima-q=<int>
                        Set the min MA component of ARIMA [default: 0]
    --max-arima-q=<int>
                        Set the max MA component of ARIMA [default: 0]
    --volume-factor=<float>
                        Set the volume weight factor [default: 0]
    --retry-count=<int>
                        Set the retry count due to API errors [default: 9]
    --dry-run           Invoke a trader with dry-run mode
    --min-span=<int>    Set the min span of EMA [default: 5]
    --max-span=<int>    Set the max span of EMA [default: 60]
    --n-splits=<int>    Set the number of splits [default: 10]
    --count=<int>       Set the number of candles [default: 5000]

Commands:
    close                   Close positions (if not <instrument>, close all)
    trade                   Trade currencies
    optimize                Optimize EMA span

Arguments:
    <instrument>            Forex instrumnt such as EUR_USD and USD_JPY
"""

import logging
import os

import v20
from docopt import docopt

from . import __version__
from .ema import optimize_ema_span
from .order import close_positions
from .trader import AutoTrader


def main():
    args = docopt(__doc__, version=f'armaf {__version__}')
    _set_log_config(debug=args['--debug'], info=args['--info'])
    logger = logging.getLogger(__name__)
    logger.debug(f'args:{os.linesep}{args}')
    oanda_account_id = (args['--oanda-account'] or os.getenv('OANDA_ACCOUNT'))
    oanda_api_token = (args['--oanda-token'] or os.getenv('OANDA_TOKEN'))
    if args.get('trade'):
        logger.info('Autonomous trading')
        AutoTrader(
            oanda_account_id=oanda_account_id, oanda_api_token=oanda_api_token,
            oanda_environment=args['--oanda-env'],
            instruments=args['<instrument>'],
            granularity=args['--granularity'],
            betting_strategy=args['--betting-strategy'],
            margin_lack_reset=args['--margin-lack-reset'],
            scanned_transaction_count=int(args['--scanned-transaction-count']),
            unit_margin_ratio=float(args['--unit-margin']),
            preserved_margin_ratio=float(args['--preserved-margin']),
            take_profit_limit_ratio=float(args['--take-profit-limit']),
            trailing_stop_limit_ratio=float(args['--trailing-stop-limit']),
            stop_loss_limit_ratio=float(args['--stop-loss-limit']),
            max_spread_ratio=float(args['--max-spread']),
            sleeping_ratio=float(args['--sleeping']),
            trigger_ssr_window=args['--trigger-ssr-window'],
            trigger_test_size=args['--trigger-test-size'],
            trigger_forecast_size=args['--trigger-forecast-size'],
            trend_ssr_window=args['--trend-ssr-window'],
            trend_test_size=args['--trend-test-size'],
            trend_forecast_size=args['--trend-forecast-size'],
            significance_level=float(args['--significance-level']),
            min_arima_p=args['--min-arima-p'],
            max_arima_p=args['--max-arima-p'],
            min_arima_d=args['--min-arima-d'],
            max_arima_d=args['--max-arima-d'],
            min_arima_q=args['--min-arima-q'],
            max_arima_q=args['--max-arima-q'],
            volume_factor=float(args['--volume-factor']),
            retry_count=int(args['--retry-count']), log_dir_path=None,
            quiet=args['--quiet'], dry_run=args['--dry-run']
        ).invoke()
    else:
        oanda_api = _create_oanda_api(
            api_token=oanda_api_token, environment=args['--oanda-env']
        )
        if args.get('close'):
            close_positions(
                api=oanda_api, account_id=oanda_account_id,
                instruments=args['<instrument>']
            )
        elif args.get('optimize'):
            optimize_ema_span(
                api=oanda_api, account_id=oanda_account_id,
                min_span=int(args['--min-span']),
                max_span=int(args['--max-span']),
                n_splits=int(args['--n-splits']),
                granularity=args['--granularity'], count=int(args['--count']),
                instruments=args['<instrument>']
            )


def _set_log_config(debug=None, info=None):
    if debug:
        lv = logging.DEBUG
    elif info:
        lv = logging.INFO
    else:
        lv = logging.WARNING
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=lv
    )


def _create_oanda_api(api_token, environment='trade', stream=False, **kwargs):
    return v20.Context(
        hostname='{0}-fx{1}.oanda.com'.format(
            ('stream' if stream else 'api'), environment
        ),
        token=api_token, **kwargs
    )
