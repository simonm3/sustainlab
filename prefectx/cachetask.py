import logging
import datetime
import os
from functools import partial
from uuid import uuid4

from prefect.futures import PrefectFuture
from prefect.tasks import *
from prefect.flows import flow, Flow
from prefect.orion.schemas.data import DataDocument
from prefect.orion.schemas.states import Completed
from prefect.context import get_run_context

from . import gcontext
from .stores import Store, Filestore
from .cache import Cache

log = logging.getLogger("prefect.prefectx")


class CacheFunc(Cache):
    """ decorator to wrap embedded function
    resolve target in CacheTask.__call__ before submission not Func.__call__ at remote execution time.
    """

    def __call__(self, *args, **kwargs):
        data = self.run(*args, **kwargs)
        result = self.get_result(data, self.target)
        return result


class CacheTask(Task):
    def __init__(self, fn, **kwargs):
        """extension to prefect.Task to cache outputs as files using target template"""
        # config parameters
        target = kwargs.pop("target", None) or fn.__name__
        store = kwargs.pop("store", True)
        self.skip = kwargs.pop("skip", True)
        self.name = kwargs.pop("name", None)

        # init then assign function to avoid functools.update_wrapper
        super().__init__(fn, **kwargs)
        self.fn = CacheFunc(fn, target=target, store=store)

    def __call__(self, *args, wait_for=None, **kwargs):
        # resolve target
        self.fn.target = self.fn.fill_template(self.fn.target, *args, **kwargs)

        # check if complete
        if self.skip:
            if os.path.isfile(self.fn.target):
                future = Skipped(self.fn.store(self.fn.target))
                return future

        # resolve name
        self.name = self.fn.fill_template(self.name, *args, **kwargs)
        future = super().__call__(*args, wait_for=wait_for, **kwargs)
        return future


class Skipped(PrefectFuture):
    """returned by tasks that skip the scheduler
    inherits methods called in flows such as wait and result
    does not define a task_runner
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
    # extra Cache
    target: str = None,
    store: Store = Filestore,
    # extra CacheTask (prefect only)
    skip: bool = True,
):
    """
    decorator to create CacheTask. This will return target if exists; load input data from Stores; save output to target; return Store(target). runs tasks on prefect2.

    :param name: task name. can be a template string
    :param target: template string for target file
    :para store: class inherited from Store. default=Filestore
    :param skip: skip the scheduler for completed tasks
    """
    if fn:
        return CacheTask(**locals())
    else:
        # enable default parameters to be set before decorator called
        del fn
        return partial(task, **locals())
