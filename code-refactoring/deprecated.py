import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

import asyncio
import json
import os
from typing import Any, Callable, Dict, List

import numpy as np
from tqdm import tqdm
from utils import call_gpt, sort_python_code


def question_to_prompt(question: str) -> Callable[[str, str], str]:
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

SIMPLIFY_PROMPT = "Please update this code to simplify it. Extract function which should be named (they are semantically meaningful on their own and distinct from their surrounding -- e.g. separate IO from computation). Inline functions which are not semantically meaningful on their own. Remove unused variables from function bodies and function signatures."

COMPREHENSIONS_PROMPT = "Please convert while/for-loops into list-comprehensions, dictionary-comprehensions, or generator-comprehensions."


REFACTORING_PIPELINE = [
    ("Rename Prompt", RENAME_VARS_AND_FUNCS_PROMPT),
    ("DRY Prompt", DRY_PROMPT),
    ("Simplify Prompt", SIMPLIFY_PROMPT),
    ("Comprehensions Prompt", COMPREHENSIONS_PROMPT),
]


def validate(code, problem_path) -> bool:
    os.makedirs("refactor-temp", exist_ok=True)
    with open("refactor-temp/all_codes.json", "w") as f:
        json.dump({"0": [code]}, f)
    with open("refactor-temp/filepaths.json", "w") as f:
        json.dump([problem_path], f)
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
        ]
    )
    try:
        with open("refactor-temp/all_results.json", "r") as f:
            body = json.load(f)
            res = np.all(body["0"])
            return res
    except:
        return False


def get_question(problem_path):
    with open(os.path.join(problem_path, "question.txt"), "r") as f:
        problem_question = f.read()
    return problem_question


def clean_up_gpt_turbo(code):
    if "```" in code:
        code = code.split("```")[1]
    if code.startswith("python"):
        code = code[6:]
    try:
        code = json.loads(code)
    except:
        pass
    return code


async def refactor(problem_path, get_prompt, code, max_tries=4) -> List[Dict[str, Any]]:
    """
    return format should be a list of individual checkpoints

    """
    checkpoints = []
    for name, prompt in REFACTORING_PIPELINE:
        checkpoint = {"name": name, "history": [code]}
        prompt = get_prompt(prompt, code)

        for _ in range(max_tries):
            new_code, success = await call_gpt(prompt)
            if not success:
                return []

            new_code = clean_up_gpt_turbo(new_code)
            checkpoint["history"].append(new_code)

            if validate(new_code, problem_path):
                code = new_code
                checkpoint["success"] = True
                break
        else:
            print(
                f"Failed to generate valid code for {name} within {max_tries} tries. Skipping the refactoring task."
            )
            checkpoint["success"] = False
        checkpoints.append(checkpoint)

    # toposort the code
    new_code = sort_python_code(code)
    checkpoints.append(
        {
            "name": "Toposort",
            "history": [code, new_code],
            "success": validate(new_code, problem_path),
        }
    )
    return checkpoints


async def refactor_code(index, code, problem_path, output_dir):
    """
    Get refactored code for the given problem.

    :param code: The code to refactor.
    :param problem_path: The path to the problem.
    :param max_tries: The maximum number of invalid refactored code to generate before raising an exception. Defaults to 4.
    """
    problem_question = get_question(problem_path)
    get_prompt = question_to_prompt(problem_question)

    id = problem_path.split("/")[-1]
    destination = os.path.join(output_dir, f"{id}/{index}.json")

    # check if it already exists
    if os.path.exists(destination):
        return

    package = {"start_code": code}

    # check if the original code is valid
    if not validate(code, problem_path):
        package.update({"end_reason": "original-invalid", "refactoring_steps": []})
    else:
        checkpoints = await refactor(problem_path, get_prompt, code)
        package.update(
            {
                "end_reason": "success" if checkpoints else "failed",
                "checkpoints": checkpoints,
            }
        )

    os.makedirs(os.path.join(output_dir, id), exist_ok=True)
    with open(destination, "w") as f:
        json.dump(package, f, indent=4)


async def main(output_dir: str, start: int = 0, end: int = float("inf")):
    training_path = "APPS/train"
    problems = sorted(os.listdir(training_path))

    # limit the number of problems to refactor
    end = min(end, len(problems))
    problems = problems[start:end]

    bar = tqdm(total=len(problems))

    async def task(problem):
        problem_path = os.path.join(training_path, problem)
        if not os.path.isdir(problem_path):
            return

        with open(os.path.join(problem_path, "solutions.json"), "r") as f:
            solutions = json.load(f)

        for i, code in enumerate(solutions[:1]):
            await refactor_code(i, code, problem_path, output_dir)
        bar.update(1)
        # display bar
        bar.refresh()
        print(f"Finished {problem_path}")

    await asyncio.gather(*[task(problem) for problem in problems])

    bar.close()


if __name__ == "__main__":
    # take in argument that is destination file path
    if len(sys.argv) < 2:
        print("Usage: python3 refactor.py <output_dir>")
        exit(1)
    output_dir = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    end = int(sys.argv[3]) if len(sys.argv) > 3 else float("inf")
    asyncio.run(main(output_dir, start, end))
