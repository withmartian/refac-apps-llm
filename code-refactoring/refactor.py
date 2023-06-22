import os
from functools import reduce
from typing import Callable

from utils import call_gpt


def question_to_prompt(question: str) -> Callable[[str, str], str]:
    """
    Get a prompt function which takes in code and returns a prompt.
    """

    def get_prompt(instructions: str, code: str) -> str:
        return f"""Here is a programming problem --
```plain text
{question}
```
Here is some python code to solve the problem --
```python
{code}
```
We are refactoring this code. {instructions}"""

    return get_prompt


RENAME_VARS_AND_FUNCS_PROMPT = "Please rename variables and functions for clarity."

DRY_PROMPT = "Please update this code to conform to DRY (don't repeat yourself) principles. Extract duplicated code into functions and turn repetitive variables into data structures."

SIMPLY_PROMPT = "Please update this code to simplify it. Extract function which should be named (they are semantically meaningful on their own and distinct from their surrounding -- e.g. separate IO from computation). Inline functions which are not semantically meaningful on their own. Remove unused variables from function bodies and function signatures."

COMPREHENSIONS_PROMPT = "Please convert while/for-loops into list-comprehensions, dictionary-comprehensions, or generator-comprehensions."


REFACTORING_PIPELINE = [
    ("Rename Prompt", RENAME_VARS_AND_FUNCS_PROMPT),
    ("DRY Prompt", DRY_PROMPT),
    ("Simplify Prompt", SIMPLY_PROMPT),
    ("Comprehensions Prompt", COMPREHENSIONS_PROMPT),
]


def validate(code, problem_path) -> bool:
    return True


def get_question(problem_path):
    with open(os.path.join(problem_path, "question.txt"), "r") as f:
        problem_question = f.read()
    return problem_question


def get_refactored_code(code, problem_path, max_tries=4):
    """
    Get refactored code for the given problem.

    :param code: The code to refactor.
    :param problem_path: The path to the problem.
    :param max_tries: The maximum number of invalid refactored code to generate before raising an exception. Defaults to 4.
    """
    if max_tries < 1:
        raise ValueError("max_tries must be at least 1.")

    problem_question = get_question(problem_path)
    get_prompt = question_to_prompt(problem_question)

    working_code = code
    for name, prompt in REFACTORING_PIPELINE:
        prompt = get_prompt(prompt, working_code)

        for _ in range(max_tries):
            new_code = call_gpt(prompt)
            if validate(new_code, problem_path):
                working_code = new_code
                break
        else:
            print(
                f"Failed to generate valid code for {name} within {max_tries} tries. Skipping the refactoring task."
            )

    return working_code
