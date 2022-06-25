""" 
reset
    sudo rm -rf ~/v1dev/storage/* ~/v1dev/awstest ~/.prefect/*
    printf '4\n/v1/storage\nv1_storage\ny' | prefect storage create

temp=/tmp/prefect
"""
"""
Is here the best place to put feedback on Orion or should that go somewhere else? My feedback so far. Really impressed with it generally. The approach to creating flows/tasks is really easy. Some issues though many of these may relate to things still under development:

make easier to extend/adapt
* task function is defined with keyword only signatures and no **kwargs. 
* possibly same with other functions that may want to extend?

make it less intrusive so tasks can be called outside flow for testing
* setting that disables @task globally so can call decorated function directly outside a flow
* getLogger function that returns get_run_logger if inside a task/flow else calls logging.getLogger
* if task called outside flow then call task.fn rather than raise exception

database
* prefect orion database reset does not work
* make database structure cleaner e.g. name of cache file is cryptic and even more cryptically stored in the database.

storage
* api that can be called via python rather than command that requires answers to prompts
* use location as default name and allow special chars e.g. /v1 or s3://simonm3
* allow set-default to accept name rather than just id
* if no default storage then just use default location without warnings

ui
* create bookmarkable view that shows flow * state * number tasks
* barchart shows only a few bars that are not labelled nor any hover so unclear. prefect1 barchart much clearer.
* refresh page does not work on some pages
* crashed tasks or stopped tasks still show as running on UI and database

docs
* move agent/work_queue from getting started to advanced usage.
* add docs for database structure
* add examples for extending/adapting task/Task etc..
* explain the UI radar view as unclear what it is supposed to show
"""
