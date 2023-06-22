from dotenv import load_dotenv

load_dotenv()

import asyncio
import json
import os
from typing import List

from utils import call_gpt

MIN_DESIRED_TEST_CASES = 10


async def generate_test_case(problem_description) -> str:
    # TODO: check if this prompt is good enough
    prompt = f"""You are a test case input generator.
Given the following problem description:
```
{problem_description}
```
What is the input that you would generate? Only include the input, not the output. Do not include any other text.
"""
    return await call_gpt(prompt)


def validate_test_case(test_case: str) -> bool:
    """
    Validate the given test case.
    """
    # TODO: import the test case validator and use it here
    ...
    # so we want the input on all the problems
    # get the output from the problem


def get_id(filepath: str) -> int:
    return int(filepath.split("/")[-1])


async def generate_test_cases(
    id: int,
    problem_description: str,
    n: int,
    old_test_cases: List[str],
    max_tries: int = 10,
) -> List[str]:
    """
    Generate n test cases for the given problem description.

    If the number of invalid test cases generated exceeds cap, exits early and returns the valid test cases.

    :param problem_description: The problem description.
    :param n: The number of test cases to generate.
    :param max_tries: The maximum number of invalid test cases to generate before raising an exception.
    """
    new_test_cases = []
    for i in range(n):
        if os.path.exists(f"generated/tests/{id}/{i}.txt"):
            with open(f"generated/tests/{id}/{i}.txt", "r") as f:
                new_test_cases.append(f.read())
            continue

        os.makedirs("generated/tests/{id}", exist_ok=True)
        for _ in range(max_tries):
            test_case = await generate_test_case(problem_description)
            if validate_test_case(test_case, old_test_cases):
                new_test_cases.append(test_case)
                break
        else:
            return new_test_cases, False

        with open(f"generated/tests/{id}/{i}.txt", "w") as f:
            f.write(test_case)
    return new_test_cases, True


def get_all_APPS_filepaths() -> List[str]:
    """
    Return a list of all filepaths in the APPS/train directory.
    """
    return [
        os.path.join("APPS/train", filename) for filename in os.listdir("APPS/train")
    ]


def get_curr_test_cases(filepath: str) -> List[str]:
    filepath = os.path.join(filepath, "input_output.json")
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r") as f:
        return json.load(f)["inputs"]


def get_problem_description(filepath: str) -> str:
    """
    Return the problem description for the given filepath.
    """
    filepath = os.path.join(filepath, "question.txt")
    with open(filepath, "r") as f:
        return f.read()


async def main():
    async def task(filepath):
        old_test_cases = get_curr_test_cases(filepath)
        num_test_cases = len(old_test_cases)
        if num_test_cases >= MIN_DESIRED_TEST_CASES:
            return

        # TODO: REMOVE THIS LATER ONCE WE CAN GEN TEST CASES WITHOUT ANY
        if num_test_cases < 1:
            return

        id = get_id(filepath)
        problem_description = get_problem_description(filepath)
        new_test_cases, success = await generate_test_cases(
            id,
            problem_description,
            MIN_DESIRED_TEST_CASES - num_test_cases,
            old_test_cases,
        )
        if not success:
            print(
                f"Failed to accumulate {MIN_DESIRED_TEST_CASES} test cases for {filepath}"
            )
        else:
            print(f"Generated {len(new_test_cases)} test cases for {filepath}")

    filepaths = get_all_APPS_filepaths()

    # TODO: REMOVE AFTER TESTING
    # NOTE: this limits the number of files to process for testing purposes
    filepaths = filepaths[:1]

    await asyncio.gather(*[task(filepath) for filepath in filepaths])


if __name__ == "__main__":
    asyncio.run(main())
