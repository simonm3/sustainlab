import logging
from functools import partial
import os
import inspect
import os
import datetime
from tqdm.auto import tqdm
from .filepath import Filepath

import defaultlog

log = logging.getLogger(__name__)

# TODO what else could be useful in target fstring?
FSTRING_MODULES = [os, datetime]


# def run(task, arglist, contextlist):
#     """ run task on arglist """
#     if not isinstance(arglist, (tuple, list)):
#         arglist = [arglist]
#     results = []
#     iterable = list(zip(arglist, contextlist))
#     for i, (args, context) in enumerate(tqdm(iterable)):
#         if not isinstance(args, (tuple, list)):
#             args = [args]
#         try:
#             Filetask.context = context
#             result = task(*args)
#             results.append(result)
#         except:
#             log.exception(f"problem with {i}")
#     return results


def f(fstring, kwargs):
    """evaluate fstring at runtime including limited set of modules"""
    modules = {module.__name__: module for module in FSTRING_MODULES}
    return eval(f"f'{fstring}'", modules, kwargs)


class Filetask:
    """decorator to cache outputs as files using target template
            populate target template using args/kwargs/taskname/context
            return target if exists
            load input data from Filepaths
            save output to target
            return Filepath(target)
    """

    # run context e.g. dict(base="some_town")
    context = dict()

    def __init__(self, fn, target=None, save=True):
        self.fn = fn
        # base can be fn args/kwargs or Filetask.context
        self.target = target or "working/{taskname}/{base}"
        self.save = save

    def __call__(self, *args, **kwargs):
        # format target at runtime using args/kwargs, taskname, context
        sig = inspect.signature(self.fn)
        bound = {
            k: v.default
            for k, v in sig.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        bound.update(sig.bind(*args, **kwargs).arguments)
        bound["taskname"] = self.fn.__name__
        bound.update(self.context)
        bound = {k: v.path if isinstance(v, Filepath) else v for k, v in bound.items()}
        bound = {k: v for k, v in bound.items() if isinstance(v, str)}
        target = f(self.target, bound)

        if os.path.exists(target):
            return Filepath(target)

        # load Filepath inputs automatically
        args = [v.load() if isinstance(v, Filepath) else v for v in args]
        kwargs = {
            k: v.load() if isinstance(v, Filepath) else v for k, v in kwargs.items()
        }

        # execute function
        data = self.fn(*args, **kwargs)

        if self.save:
            # save data and return Filepath so it can be loaded automatically downstream
            return Filepath(target, data)
        else:
            # neither save nor load automatically
            return target

def filetask(fn=None, **kwargs):
    if fn:
        return Filetask(fn, **kwargs)
    else:
        return partial(filetask, **kwargs)

