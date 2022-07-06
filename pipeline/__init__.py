import os
from prefect.client import get_client

# extra_loggers logs to orion
os.environ.update(PREFECT_LOGGING_EXTRA_LOGGERS="pipeline")


async def limits():
    try:
        async with get_client() as client:
            res = await client.delete_concurrency_limit_by_tag("sentence")
    except:
        pass
    async with get_client() as client:
        limit_id = await client.create_concurrency_limit(
            tag="sentence", concurrency_limit=1
        )
    async with get_client() as client:
        res = await client.read_concurrency_limits(100, 0)
    return await res


if __name__ == "__main__":
    res = limits()
    print(res)
