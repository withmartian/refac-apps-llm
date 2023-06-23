import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

import asyncio
import json
import os
from collections import defaultdict
from typing import Callable

from tqdm import tqdm
from utils import call_gpt

import numpy as np


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


# def reorganize_methods(code):
#     pattern = r"^def\s*\w*\(.*"
#     splitted = re.split(pattern, code)
#     functions = re.findall(pattern, code)
#     res = splitted[0]
#     splitted = splitted[1:]

#     assert len(splitted) == len(functions)

#     for i in range(len(splitted)):
#         # count space in beginning of next line
#         num_spaces = len(splitted[i]) - len(splitted[i].lstrip())
#         # remove spaces from each one of the lines
#         splitted[i] = "\n".join([s[num_spaces:] for s in splitted[i].split("\n")])
#         splitted[i] = reorganize_methods(splitted[i])
#         # add spaces back
#         splitted[i] = "\n".join([" " * num_spaces + s for s in splitted[i].split("\n")])

#     # topo sort based on function calls
#     edges = defaultdict(list)
#     print(functions)
#     print(edges)

#     name_regex = r"def\s+(\w+)\s*\("
#     function_names = [re.findall(name_regex, f)[0] for f in functions]
#     for i in range(len(functions)):
#         for j in range(i + 1, len(functions)):
#             if function_names[i] + "(" in splitted[j]:
#                 edges[i].append(j)
#             if function_names[j] + "(" in splitted[i]:
#                 edges[j].append(i)

#     # toposort the edges
#     visited = set()
#     order = []

#     def dfs(node):
#         if node in visited:
#             return
#         visited.add(node)
#         for child in edges[node]:
#             dfs(child)
#         order.append(node)

#     for node in functions:
#         dfs(node)

#     for i in reversed(order):
#         res += functions[i] + splitted[i]

#     return res


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


async def refactor(problem_path, get_prompt, working_code, max_tries=4):
    checkpoints = defaultdict(list)
    step = 0
    for name, prompt in REFACTORING_PIPELINE:
        prompt = get_prompt(prompt, working_code)
        starting_code = working_code
        new_code_history = []

        for _ in range(max_tries):
            new_code, success = await call_gpt(prompt)
            if not success:
                return {}
            new_code_history.append(new_code)

            if validate(new_code, problem_path):
                working_code = new_code
                break
        else:
            print(
                f"Failed to generate valid code for {name} within {max_tries} tries. Skipping the refactoring task."
            )

        checkpoints[name].append(
            {
                "prompt": prompt,
                "old_code": starting_code,
                "new_code": new_code_history,
                "step": step,
                "success": working_code == new_code,
            }
        )
        step += 1
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

    # try it first
    if not validate(code, problem_path):
        package = {
            "original_worked": False,
            "gpt-failure": False,
            "refactoring_steps": [],
        }
    elif not (checkpoints := await refactor(problem_path, get_prompt, code)):
        package = {
            "original_worked": True,
            "gpt-failure": True,
            "refactoring_steps": [],
        }
    else:
        package = {
            "original_worked": True,
            "gpt-failure": False,
            "refactoring_steps": checkpoints,
        }

    os.makedirs(os.path.join(output_dir, id), exist_ok=True)
    with open(destination, "w") as f:
        json.dump(package, f, indent=4)


async def main(output_dir: str):
    training_path = "APPS/train"
    problems = sorted(os.listdir(training_path))

    # TODO: remove this restriction after testing
    problems = problems[:1]
    bar = tqdm(total=len(problems))

    async def task(problem):
        problem_path = os.path.join(training_path, problem)
        if not os.path.isdir(problem_path):
            return

        with open(os.path.join(problem_path, "solutions.json"), "r") as f:
            solutions = json.load(f)

        for i, code in enumerate(solutions):
            await refactor_code(i, code, problem_path, output_dir)
        bar.update(1)
        # display bar
        bar.refresh()
        print(f"Finished {problem_path}")

    await asyncio.gather(*[task(problem) for problem in problems])

    bar.close()


if __name__ == "__main__":
    # take in argument that is destination file path
    if len(sys.argv) != 2:
        print("Usage: python3 refactor.py <output_dir>")
        exit(1)
    output_dir = sys.argv[1]
    asyncio.run(main(output_dir))


# example_scripts = [
#     """
# def is_prime(number):
#     if number < 2:
#         return False
#     for i in range(2, number):
#         if number % i == 0:
#             return False
#     return True

# def print_prime_numbers(n):
#     primes = [p for p in range(2, n + 1) if is_prime(p)]
#     print(primes)

# print_prime_numbers(50)
# """,
#     """
# def fibonacci_recursive(n):
#     if n == 0 or n == 1:
#         return n
#     return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

# def print_fibonacci_recursive(n):
#     print(f"The {n}th Fibonacci number is {fibonacci_recursive(n)}")

# print_fibonacci_recursive(10)
# """,
#     """

# from random import randint

# def generate_random_list(size):
#     result = [randint(1, 100) for _ in range(size)]
#     return result

# def random_sum(size):
#     numbers = generate_random_list(size)
#     return sum(numbers)

# print(random_sum(10))
# """,
#     """
# def reverse_string(s):
#     return s[::-1]

# def reverse_and_print(s):
#     print(reverse_string(s))

# reverse_and_print("Example string")
# """,
#     """
# def greet(name):
#     return "Hello, " + name + "!"

# def greet_and_print(name):
#     print(greet(name))

# greet_and_print("John")
# """,
#     """
# def main_function():
#     def helper_function():
#         print("This is the helper function")

#     helper_function()
#     print("This is the main function")

# main_function()
# """,
#     """
# def recursive_function(n):
#     if n > 0:
#         print(n)
#         recursive_function(n-1)

# recursive_function(5)
# """,
#     """
# def outer_function():
#     print("This is the outer function")

#     def inner_function_1():
#         print("This is the inner function 1")

#     def inner_function_2():
#         inner_function_1()
#         print("This is the inner function 2")

#     inner_function_2()

# outer_function()
# """,
# ]

# for script in example_scripts:
#     print(script)
#     input()
#     print(reorganize_methods(script))
#     print()
#     input()
