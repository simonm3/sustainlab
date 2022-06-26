import logging
import datetime
import functools
import inspect
import os
from functools import partial
from uuid import uuid4

from prefect.context import get_run_context
from prefect.futures import PrefectFuture
from prefect.tasks import *
from prefect.flows import flow, Flow
from prefect.orion.schemas.data import DataDocument
from prefect.orion.schemas.states import Completed
from prefect.task_runners import SequentialTaskRunner, ConcurrentTaskRunner

from .filepath import use_filepaths, Filepath

log = logging.getLogger(__name__)

# TODO what else could be useful in target fstring?
FSTRING_MODULES = [os, datetime]


def f(fstring, kwargs):
    """evaluate fstring at runtime including limited set of modules"""
    modules = {module.__name__: module for module in FSTRING_MODULES}
    return eval(f"f'{fstring}'", modules, kwargs)


class Task(Task):
    """override Task to behave like a makefile
    fill target template at runtime
    skip task if target exists
    wrap task.fn to load Filepath inputs; save output; return Filepath

    Usage:
           from prefectx.makefile import task
           @task(target="some/path/{template}")
           def myfunc(**kwargs):
               # do stuff
               return x

    :param kwargs: as parent plus below
    :param target: fstring template to save return data. evaluated at runtime including modules [os, datetime]
    :param save: set to False if function saves own output. default:True.
    """

    def __init__(self, **kwargs):
        self.target = kwargs.pop("target", kwargs["fn"].__name__)
        self.save = kwargs.pop("save", True)

        super().__init__(**kwargs)

    def __call__(self, *args, wait_for=None, **kwargs):
        # fill target template using args/kwargs, upstream target paths, taskname, flow_run parameters
        sig = inspect.signature(self.fn)
        bound = sig.bind(*args, **kwargs).arguments
        bound = {
            k: str(v.target) if isinstance(v, PrefectFuture) else v
            for k, v in bound.items()
        }
        bound["taskname"] = self.fn.__name__
        bound.update(get_run_context().flow_run.parameters)
        target = f(self.target, bound)

        if os.path.isfile(target):
            # skip completed without even contacting the scheduler
            # log.info(f"{self.fn.__name__} skipped as target exists {target}")
            future = Skipped(Filepath(target))
        else:
            # wrap self.fn before submit to orion
            unwrapped = self.fn
            self.fn = use_filepaths(self.fn, target, save=self.save)
            future = super().__call__(*args, wait_for=wait_for, **kwargs)
            self.fn = unwrapped

        # enable upstream target paths to be used to fill downstream target path template
        future.target = target
        return future


class Skipped(PrefectFuture):
    """returned by tasks that skip the scheduler
    ensures return value can be treated the same e.g. wait and result methods
    but does not set a task_runner etc..
    """

    def __init__(self, data):
        self.run_id = uuid4()
        self.asynchronous = False
        self._final_state = Completed(
            data=DataDocument.encode(encoding="cloudpickle", data=data)
        )
        self.target = None

    def get_state(self):
        return self._final_state

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"Skipped({self.target})"

    def _ipython_display_(self):
        print(repr(self))


## ovreride function decorators ######################################################


def task(
    fn=None,
    name: str = None,
    description: str = None,
    tags: Iterable[str] = None,
    cache_key_fn: Callable[["TaskRunContext", Dict[str, Any]], Optional[str]] = None,
    cache_expiration: datetime.timedelta = None,
    retries: int = 0,
    retry_delay_seconds: Union[float, int] = 0,
    save: bool = True,
    target: str = None,
):
    """ adds save and target parameters """
    if fn:
        return Task(**locals())
    else:
        return partial(task, **locals())


flow_ = flow


def flow(*args, **kwargs):
    """wrap flow to convert returned skipped to states"""
    runner = kwargs.get("task_runner", ConcurrentTaskRunner)
    if isinstance(runner, ConcurrentTaskRunner):
        # concurrent has single version of Task so cannot handle multiple targets simultaneously
        raise Exception(
            "prefectx does not support ConcurrentTaskRunner. Use Sequential, Dask or Ray"
        )
    flow1 = flow_(*args, **kwargs)
    if isinstance(flow1, Flow):
        # wrap flow to return state
        flow1.fn = newflow(flow1.fn)

    return flow1


def newflow(func):
    """replace returned Skippeds with states"""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        ret = func(*args, **kwargs)
        if not isinstance(ret, (list, tuple)):
            ret = [ret]
        ret = [s.get_state() if isinstance(s, Skipped) else s for s in ret]
        return ret

    return wrapped
