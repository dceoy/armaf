#!/usr/bin/env python

import logging
import os
from pprint import pformat


def close_positions(api, account_id, instruments=None):
    logger = logging.getLogger(__name__)
    logger.info("Position closing")
    if not account_id:
        raise ValueError("account ID required")
    else:
        pos_res = api.position.list_open(accountID=account_id)
    _log_response(pos_res, logger=logger)
    positions = pos_res.body["positions"]
    insts_to_close = list()
    if positions:
        for p in positions:
            if not instruments or p.instrument in instruments:
                insts_to_close.append(p.instrument)
                pos = {
                    "instrument": p.instrument,
                    **{
                        f"{k}Units": (
                            "NONE" if int(getattr(p, k).units) == 0 else "ALL"
                        )
                        for k in ["long", "short"]
                    },
                }
                logger.debug(f"pos:\t{pos}")
                res = api.position.close(accountID=account_id, **pos)
                _log_response(res, logger=logger)
                if 100 <= res.status <= 399:
                    logger.debug(res.body)
                else:
                    raise RuntimeError(
                        "unexpected response:" + os.linesep + pformat(res.body)
                    )
    if insts_to_close:
        print("Positions closed:\t" + ", ".join(insts_to_close))
    else:
        print("No positions to close.")


def _log_response(response, logger=None, expected_status_range=(100, 399)):
    logger = logger or logging.getLogger(__name__)
    res_str = "response =>" + os.linesep + pformat(vars(response))
    esr = sorted(expected_status_range)
    if esr[0] <= response.status <= esr[-1]:
        logger.debug(res_str)
    else:
        logger.error(res_str)
