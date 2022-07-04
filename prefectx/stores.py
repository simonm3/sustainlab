import logging
import os
import pickle

log = logging.getLogger(__name__)


class Store:
    """location to store data
    e.g. pickle file, database
    """

    def __init__(self, loc, data=None):
        self.loc = loc
        if data is not None:
            self.save(data)

    def save(self, data):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def __str__(self):
        return self.loc

    def __repr__(self):
        return f"{self.__class__.__name__}({self.loc})"

    def _ipython_display_(self):
        print(repr(self))


class Filestore(Store):
    """file location to store data"""

    def save(self, data):
        os.makedirs(os.path.dirname(os.path.abspath(self.loc)), exist_ok=True)
        pickle.dump(data, open(self.loc, "wb"))

    def load(self):
        return pickle.load(open(self.loc, "rb"))

