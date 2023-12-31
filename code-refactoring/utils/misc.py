import json
import os
from typing import Any, Callable, Dict, List, Union

############################################
############### MISC METHODS ###############
############################################


async def cache_wrapper(path: str, func: Callable, *args, **kwargs) -> Any:
    """
    Caches the result of the function in the given path.

    :param path: The path to the cache file.
    :param main: The function to call if the cache file does not exist.
    :param args: The arguments to pass to the function.
    :param kwargs: The keyword arguments to pass to the function.
    :return: The result of the function.
    """

    is_json = path.endswith(".json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f) if is_json else f.read()
    else:
        res = await func(*args, **kwargs)
        with open(path, "w") as f:
            if is_json:
                json.dump(res, f, indent=4)
            else:
                f.write(res)
        return res


def get_json_with_default(path: str, default=lambda: dict()) -> Union[Dict, List]:
    """
    Gets the JSON from the given path, or returns the default if JSON is not
    properly formatted or the file does not exist.

    :param path: The path to the JSON file.
    :param default: Lambda function to call for the default value. (Optional, defaults to lambda: dict())
    :return: The JSON.
    """
    if not os.path.exists(path):
        return default()

    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.decoder.JSONDecodeError:
        return default()
