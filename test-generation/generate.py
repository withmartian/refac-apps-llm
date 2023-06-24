from collections import Counter
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

import asyncio
import json
import os
from typing import Any, List, Optional, Tuple

from utils import call_gpt

MIN_DESIRED_TEST_CASES = 10
MAX_TRIES = 3


async def generate_tc_input(problem_description, prior_test_cases) -> Optional[str]:
    inputs = "\n".join(repr(input) for input, _ in prior_test_cases)
    prompt = f"""You are a test case input generator.
Given the following problem description:
---
{problem_description}
---
What is the input that you would generate? Only include the input, not the output. Do not include any other text.
Some example inputs separated by new lines:
---
{inputs}
---
Make sure to provide an input not in the list above and use proper formatting:
"""
    return await call_gpt(prompt)


def get_solutions(filepath):
    with open(os.path.join(filepath, "solutions.json"), "r") as f:
        return json.load(f)


def get_outputs(solutions: List[str], tc_input: str, filepath: str) -> List[Any]:
    os.makedirs("generate-temp", exist_ok=True)

    # fn_name...
    with open("generate-temp/filepaths.json", "w") as f:
        json.dump([""], f)

    # open original test cases
    try:
        with open(os.path.join(filepath, "input_output.json"), "r") as f:
            data = json.load(f)
    except:
        data = {}

    # dump the actual test cases
    with open("generate-temp/input_output.json", "w") as f:
        data["inputs"] = [tc_input]
        data["outputs"] = [""]
        json.dump(data, f)

    # dump the code
    with open("generate-temp/all_codes.json", "w") as f:
        json.dump({"0": solutions}, f)

    subprocess.run(
        [
            "python3",
            "apps/eval/test_one_solution.py",
            "-t",
            "generate-temp/filepaths.json",
            "-r",
            "generate-temp",
            "--save",
            "generate-temp",
        ]
    )
    try:
        with open("generate-temp/the_results.json", "r") as f:
            return json.load(f)
    except:
        return []


def get_shared_output(outputs: List[Any]) -> Optional[str]:
    if len(outputs) < 1:
        return None

    # get outputs
    l = len(outputs)
    d = dict()
    for output in outputs:
        if output is not None:
            d[repr(output)] = output
    values = [repr(output) for output in outputs if output is not None]

    if not values:
        return None

    # get frequency of each output
    freq = Counter(values)
    most_common_item, most_common_freq = freq.most_common(1)[0]

    if most_common_freq >= l / 2:
        return d[most_common_item]
    return None


def generate_tc_output(test_case: str, filepath: str) -> Optional[str]:
    """
    Validate the given test case.
    """
    # get existing solutions
    solutions = get_solutions(filepath)

    # generate outputs
    outputs = get_outputs(solutions, test_case, filepath)

    # get common output if it exists
    common_output = get_shared_output(outputs)

    return common_output


def get_curr_test_cases(filepath: str) -> List[Tuple[str, str]]:
    filepath = os.path.join(filepath, "input_output.json")
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            return list(zip(data["inputs"], data["outputs"]))
    except:
        return []


def get_problem_description(filepath: str) -> str:
    """
    Return the problem description for the given filepath.
    """
    filepath = os.path.join(filepath, "question.txt")
    with open(filepath, "r") as f:
        return f.read()


def try_parse_json(s: str) -> Any:
    try:
        return json.loads(s)
    except:
        return s


async def generate_test_cases(filepath, output_dir) -> List[str]:
    """
    Generate n test cases for the given problem description.

    If the number of invalid test cases generated exceeds cap, exits early and returns the valid test cases.
    """
    id: str = filepath.split("/")[-1]
    start_path = os.path.join(output_dir, id)
    if os.path.exists(os.path.join(start_path, "marker.txt")):
        print(f"Test cases for {id} already generated.")
        try:
            with open(os.path.join(start_path, "inputs_outputs.json"), "r") as f:
                data = json.load(f)
                return list(zip(data["inputs"], data["outputs"]))
        except:
            with open(os.path.join(start_path, "marker.txt"), "w") as f:
                f.write("struggled to open inputs_outputs.json")
            return []

    try:
        problem_description = get_problem_description(filepath)
    except:
        print("Failed to get problem description.")
        # dump marker
        os.makedirs(start_path, exist_ok=True)
        with open(os.path.join(start_path, "marker.txt"), "w") as f:
            f.write("FAILED TO GET PROBLEM DESCRIPTION")
        return []
    test_cases: List[Tuple[str, str]] = get_curr_test_cases(filepath)
    start_num = len(test_cases)

    print("prior test cases: ", test_cases)

    while len(test_cases) < MIN_DESIRED_TEST_CASES:
        for _ in range(MAX_TRIES):
            # generate a test case
            tc_input = await generate_tc_input(problem_description, test_cases)
            if tc_input is None:
                print("Failed to generate test case input.")
                continue

            print("tc_input: ", tc_input)
            tc_input = try_parse_json(tc_input)

            tc_output = generate_tc_output(tc_input, filepath)
            if tc_output is None:
                # NOTE: likely won't be uncommon
                continue

            print("tc_output: ", tc_output)
            tc_output = try_parse_json(tc_output)

            # found a valid test case
            test_cases.append((tc_input, tc_output))
            break
        else:
            # failed to generate a valid test case after X tries so stop trying
            print(f"Failed to generate enough valid test cases for {filepath}.")
            print(f"Started with {start_num} test cases.")
            break

    # dump the current test cases
    os.makedirs(start_path, exist_ok=True)
    with open(os.path.join(start_path, "inputs_outputs.json"), "w") as f:
        body = {
            "input": [tc[0] for tc in test_cases],
            "output": [tc[1] for tc in test_cases],
        }
        print("body: ", body)
        json.dump(body, f, indent=4)

    # dump marker
    with open(os.path.join(start_path, "marker.txt"), "w") as f:
        f.write("MARKER")

    return test_cases


def get_all_APPS_filepaths() -> List[str]:
    """
    Return a list of all filepaths in the APPS/train directory.
    """
    return [
        os.path.join("APPS/train", filename)
        for filename in sorted(os.listdir("APPS/train"))
    ]


async def main(output_dir, start=0, end=float("inf")):
    async def task(filepath):
        test_cases = await generate_test_cases(filepath, output_dir)
        if len(test_cases) < MIN_DESIRED_TEST_CASES:
            print(
                f"Failed to accumulate {MIN_DESIRED_TEST_CASES} test cases for {filepath}"
            )
            print(f"Only {len(test_cases)} test cases were generated for {filepath}")
        else:
            print(
                f"Successfully accumulated {MIN_DESIRED_TEST_CASES} test cases for {filepath}"
            )

    filepaths = get_all_APPS_filepaths()
    end = min(end, len(filepaths))

    # limit to a subset of the filepaths
    filepaths = filepaths[start:end]
    await asyncio.gather(*[task(filepath) for filepath in filepaths])


if __name__ == "__main__":
    # take in argument that is destination file path
    if len(sys.argv) < 2:
        print("Usage: python3 generate_test_cases.py <output_dir> (<start>) (<end>)")
        exit(1)
    output_dir = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    end = int(sys.argv[3]) if len(sys.argv) > 3 else float("inf")
    asyncio.run(main(output_dir, start, end))
