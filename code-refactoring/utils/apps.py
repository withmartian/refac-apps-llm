############################################
###### APPS DATASET SPECIFIC METHODS #######
############################################


import json
import os
import subprocess

import numpy as np


def get_problem_question(problem_path: str) -> str:
    """
    Gets the question for the problem.

    :param problem_path: The path to the problem.
    :return: The question.
    """

    with open(os.path.join(problem_path, "question.txt"), "r") as f:
        problem_question = f.read()
    return problem_question


def validate_code(code: str, problem_path: str, display_stdout: bool = False) -> bool:
    """
    Checks if the code is valid for the problem.

    :param code: The code to check.
    :param problem_path: The path to the problem.
    :param display_stdout: Whether to display the stdout of the evaluation script. (Optional, defaults to False)
    :return: Whether the code is valid.
    """

    stdout = subprocess.DEVNULL if not display_stdout else None

    # prepare the files necessary for the evaluation script
    os.makedirs("refactor-temp", exist_ok=True)
    if os.path.exists("refactor-temp/all_results.json"):
        os.remove("refactor-temp/all_results.json")
    with open("refactor-temp/all_codes.json", "w") as f:
        json.dump({"0": [code]}, f)
    with open("refactor-temp/filepaths.json", "w") as f:
        json.dump([problem_path], f)

    # run the evaluation script
    subprocess.run(
        [
            "python3",
            "apps/eval/test_one_solution.py",
            "-t",
            "refactor-temp/filepaths.json",
            "-r",
            "",
            "--save",
            "refactor-temp",
        ],
        stdout=stdout,
        stderr=stdout,
    )

    # check the results
    try:
        with open("refactor-temp/all_results.json", "r") as f:
            body = json.load(f)
        return isinstance(body["0"][0][0], bool) and np.all(body["0"])
    except Exception as e:
        print(f"Error with validate results: {e}")
        return False
