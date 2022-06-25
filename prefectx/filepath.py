import logging
import functools
import os
import pickle

log = logging.getLogger(__name__)


def use_filepaths(func, target, save=True):
    """decorator to load input data from Filepaths; save output; return Filepath(target)"""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):

        # already exists
        if os.path.exists(target):
            return Filepath(target)

        # load Filepath inputs automatically
        args = [v.load() if isinstance(v, Filepath) else v for v in args]
        kwargs = {k: v.load() if isinstance(v, Filepath) else v for k, v in kwargs}

        # execute function
        data = func(*args, **kwargs)

        if save:
            # save data and return Filepath so it can be loaded automatically downstream
            return Filepath(target, data)
        else:
            # neither save nor load automatically
            return target

    return wrapped


class Filepath:
    """wrapper for a path to data"""

    def __init__(self, path, data=None):
        self.path = path
        if data is not None:
            self.save(data)

    def save(self, data):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        log.info(f"saving {self.path}")
        pickle.dump(data, open(self.path, "wb"))

    def load(self):
        return pickle.load(open(self.path, "rb"))

    def __str__(self):
        return self.path

    def __repr__(self):
        return f"Filepath({self.path})"

    def _ipython_display_(self):
        print(repr(self))
