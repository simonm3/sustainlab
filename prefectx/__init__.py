from time import sleep
import subprocess
from threading import Thread
import json
from cloudpickle import pickle
import base64


# TODO remove when fixed. likely there should be a timeout setting somewhere.
def keepalive():
    """poll the orion server to stop connection timeout"""

    def target():
        while True:
            subprocess.Popen("prefect storage ls >/dev/null", shell=True)
            sleep(100)

    Thread(target=target, daemon=True).start()


def load_cache(filename):
    """load file from cache"""
    res = json.load(open(filename))
    blob = res["blob"]
    return pickle.loads(base64.b64decode(blob))
