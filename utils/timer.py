"""
    Calculate the elapsed time of the code. Usage:
    with Timer('stuff', logger):
        do
        some
        stuff
        here
"""

import time


class Timer(object):
    def __init__(self, name, logger=None):
        self.name = name
        self.logger = logger

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.logger is not None:
            self.logger.info("%s runs for %s", self.name, (time.time() - self.tstart))
