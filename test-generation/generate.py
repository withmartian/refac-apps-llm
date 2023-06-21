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


async def generate_test_cases(
    problem_description: str, n: int, max_tries: int = 10
) -> List[str]:
    """
    Generate n test cases for the given problem description.

    If the number of invalid test cases generated exceeds cap, exits early and returns the valid test cases.

    :param problem_description: The problem description.
    :param n: The number of test cases to generate.
    :param max_tries: The maximum number of invalid test cases to generate before raising an exception.
    """
    test_cases = []
    reset_counter = 0
    while len(test_cases) < n:
        test_case = await generate_test_case(problem_description)
        if validate_test_case(test_case):
            reset_counter = 0
            test_cases.append(test_case)
        else:
            reset_counter += 1
            if reset_counter >= max_tries:
                return test_cases, False
    return test_cases, True


def get_all_APPS_filepaths() -> List[str]:
    """
    Return a list of all filepaths in the APPS/train directory.
    """
    return [
        os.path.join("APPS/train", filename) for filename in os.listdir("APPS/train")
    ]


def get_num_test_cases(filepath: str) -> int:
    """
    Return the number of existing test cases for a given filepath's problem.
    """
    filepath = os.path.join(filepath, "input_output.json")
    if not os.path.exists(filepath):
        return 0
    with open(filepath, "r") as f:
        return len(json.load(f)["inputs"])


def get_problem_description(filepath: str) -> str:
    """
    Return the problem description for the given filepath.
    """
    filepath = os.path.join(filepath, "question.txt")
    with open(filepath, "r") as f:
        return f.read()


async def main():
    async def task(filepath):
        num_test_cases = get_num_test_cases(filepath)
        if num_test_cases >= MIN_DESIRED_TEST_CASES:
            return

        problem_description = get_problem_description(filepath)
        test_cases, success = await generate_test_cases(
            problem_description, MIN_DESIRED_TEST_CASES - num_test_cases
        )
        if not success:
            print(
                f"Failed to accumulate {MIN_DESIRED_TEST_CASES} test cases for {filepath}"
            )
        else:
            print(f"Generated {len(test_cases)} test cases for {filepath}")
            print(test_cases)

    # iterate through "APPS/" directory
    filepaths = get_all_APPS_filepaths()

    # TODO: REMOVE AFTER TESTING
    # NOTE: this limits the number of files to process for testing purposes
    filepaths = filepaths[:1]

    await asyncio.gather(*[task(filepath) for filepath in filepaths])


if __name__ == "__main__":
    asyncio.run(main())
