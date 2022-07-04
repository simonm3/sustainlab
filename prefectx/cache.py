import logging
from functools import partial
import os
import inspect
import os
import datetime
from . import gcontext
from .stores import Filestore, Store

log = logging.getLogger("prefect.prefectx")


# TODO what else could be useful in target fstring?
FSTRING_MODULES = [os, datetime]


def f(fstring, kwargs):
    """evaluate fstring at runtime including limited set of modules"""
    modules = {module.__name__: module for module in FSTRING_MODULES}
    return eval(f"f'{fstring}'", modules, kwargs)


class Cache:
    """
    class decorator to wrap function in cache.
    return target if exists; load inputs from Stores; save output to target; return Store(target)

    :param target: template string for target file
    :para store: what to return. default=FileStore. None=raw data.
    """

    def __init__(self, fn, target=None, store=Filestore):
        self.fn = fn
        # base can be from args/kwargs or context
        self.target = target or "working/{taskname}/{base}"
        self.store = store

    def __call__(self, *args, **kwargs):
        target = self.fill_template(self.target, *args, **kwargs)
        if os.path.exists(target):
            return self.store(target)
        data = self.run(*args, **kwargs)
        result = self.get_result(data, target)
        return result

    def fill_template(self, template, *args, **kwargs):
        """fill template using args/kwargs, taskname, context"""
        sig = inspect.signature(self.fn)
        context = {
            k: v.default
            for k, v in sig.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        arguments = sig.bind(*args, **kwargs).arguments
        context.update(**arguments)
        context = {k: str(v) for k, v in context.items()}
        context.update(taskname=self.fn.__name__)
        context.update(gcontext)
        target = f(template, context)

        return target

    def run(self, *args, **kwargs):
        # load Store inputs automatically
        args = [v.load() if isinstance(v, Store) else v for v in args]
        kwargs = {k: v.load() if isinstance(v, Store) else v for k, v in kwargs.items()}

        # execute function
        data = self.fn(*args, **kwargs)

        return data

    def get_result(self, data, target):
        if self.store is None:
            return target
        return self.store(target, data)


def task(fn=None, target: str = None, store: Store = Filestore, **kwargs):
    """
    decorator to wrap function in cache.
    return target if exists; load inputs from Stores; save output to target; return Store(target)

    :param target: template string for target file
    :para store: what to return. default=FileStore. None=raw data.
    """
    if fn:
        del kwargs
        return Cache(**locals())
    else:
        # enable default parameters to be set before decorator called
        del fn
        del kwargs
        return partial(task, **locals())
