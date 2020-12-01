# -*- coding: utf-8 -*-
"""Module with utility functions which are *not* based on the Dataiku API"""

import logging
import functools
from typing import Callable, AnyStr
from time import perf_counter


def time_logging(log_message: AnyStr):
    """Decorator to log timing with a custom message"""

    def inner_function(function: Callable):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            start = perf_counter()
            logging.info(log_message + "...")
            value = function(*args, **kwargs)
            end = perf_counter()
            logging.info(log_message + f": done in {end - start:.2f} seconds")
            return value

        return wrapper

    return inner_function
