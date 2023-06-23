import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

import asyncio
import json
import os
from typing import List, Optional, Tuple

from utils import call_gpt

MIN_DESIRED_TEST_CASES = 10
MAX_TRIES = 3


async def generate_tc_input(problem_description, prior_test_cases) -> Optional[str]:
    # TODO: check if this prompt is good enough
    prompt = f"""You are a test case input generator.
Given the following problem description:
```
{problem_description}
```
What is the input that you would generate? Only include the input, not the output. Do not include any other text.
Some past test cases are (in JSON list format):
{json.dumps(prior_test_cases, indent=4)}
---
Make sure to provide an input not in the list above and use proper formatting:
"""
    return await call_gpt(prompt)


def get_solutions(filepath):
    with open(os.path.join(filepath, "solutions.json"), "r") as f:
        return json.load(f)


def get_output(code: str, tc_input: str, filepath: str) -> List[str]:
    # TODO:
    # temps/filepaths.json -- ["gotta-pick-the-filepath"]
    # <>
    # save
    # <>
    # temps/all_codes.json -- {"0": ["solution1", "solution2"]}
    # <>
    os.makedirs("temp", exist_ok=True)

    # subprocess.run(
    #     [
    #         "python3",
    #         "../appss/eval/test_one_solution.py",
    #         "-t",
    #         "temps/filepaths.json",
    #         "-r",
    #         "",
    #         "--save",
    #         "temps",
    #     ]
    # )
    try:
        with open("temps/all_results.json", "r") as f:
            return json.load(f)["0"]
    except:
        pass
    return []


def get_shared_output(outputs: List[List[str]]):
    if len(outputs) < 1:
        return None

    first_output = outputs[0]
    for possible in first_output:
        for output in outputs:
            if possible not in output:
                break
        else:
            # we found a valid output
            return possible
    return None


def generate_tc_output(test_case: str, filepath: str) -> Optional[str]:
    """
    Validate the given test case.
    """
    # get existing solutions
    solutions = get_solutions(filepath)

    outputs = []
    for code in solutions:
        # get the possible outputs
        outputs.append(get_output(code, test_case, filepath))

    # the ith output is a list of possible formats of the output
    return get_shared_output(outputs)


def get_curr_test_cases(filepath: str) -> List[Tuple[str, str]]:
    filepath = os.path.join(filepath, "input_output.json")
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r") as f:
        data = json.load(f)
        return list(zip(data["inputs"], data["outputs"]))


def get_problem_description(filepath: str) -> str:
    """
    Return the problem description for the given filepath.
    """
    filepath = os.path.join(filepath, "question.txt")
    with open(filepath, "r") as f:
        return f.read()


async def generate_test_cases(filepath, output_dir) -> List[str]:
    """
    Generate n test cases for the given problem description.

    If the number of invalid test cases generated exceeds cap, exits early and returns the valid test cases.
    """
    problem_description = get_problem_description(filepath)
    test_cases: List[Tuple[str, str]] = get_curr_test_cases(filepath)
    start_num = len(test_cases)

    while len(test_cases) < MIN_DESIRED_TEST_CASES:
        for _ in range(MAX_TRIES):
            # generate a test case
            tc_input = await generate_tc_input(problem_description, test_cases)
            if tc_input is None:
                continue
            tc_output = generate_tc_output(tc_input, filepath)

            # if the test case is valid, add it to the list
            if tc_output is not None:
                test_cases.append((tc_input, tc_output))
                break
        else:
            print(f"Failed to generate enough valid test cases for {filepath}.")
            print(f"Started with {start_num} test cases.")
            break

    # dump the current test cases
    id: str = filepath.split("/")[-1]
    start_path = os.path.join(output_dir, id)
    os.makedirs(start_path, exist_ok=True)
    with open(os.path.join(start_path, "inputs_outputs.json"), "w") as f:
        body = {
            "input": [tc[0] for tc in test_cases],
            "output": [tc[1] for tc in test_cases],
        }
        json.dump(body, f, indent=4)

    return test_cases


def get_all_APPS_filepaths() -> List[str]:
    """
    Return a list of all filepaths in the APPS/train directory.
    """
    return [
        os.path.join("APPS/train", filename) for filename in os.listdir("APPS/train")
    ]


async def main(output_dir):
    async def task(filepath):
        test_cases = await generate_test_cases(filepath, output_dir)
        if len(test_cases) < MIN_DESIRED_TEST_CASES:
            print(
                f"Failed to accumulate {MIN_DESIRED_TEST_CASES} test cases for {filepath}"
            )
            print(f"Only {len(test_cases)} test cases were generated for {filepath}")

    filepaths = get_all_APPS_filepaths()

    # TODO: REMOVE AFTER TESTING
    # NOTE: this limits the number of files to process for testing purposes
    filepaths = filepaths[:1]

    await asyncio.gather(*[task(filepath) for filepath in filepaths])


if __name__ == "__main__":
    # take in argument that is destination file path
    if len(sys.argv) != 2:
        print("Usage: python3 generate_test_cases.py <output_dir>")
        exit(1)
    output_dir = sys.argv[1]
    asyncio.run(main(output_dir))
