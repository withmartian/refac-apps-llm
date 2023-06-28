import argparse
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

import asyncio
import json
import os
from typing import Dict, List

import numpy as np
from tqdm import tqdm
from utils import call_gpt_directly, sort_python_code


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


def parse_code(code):
    if "```" in code:
        code = code.split("```")[1]
    if code.startswith("python"):
        code = code[6:]
    return code


async def cache_wrapper(path, func, *args, **kwargs):
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    else:
        res = await func(*args, **kwargs)
        if res is None:
            return None
        with open(path, "w") as f:
            f.write(res)
        return res


async def get_code_smells(code, problem_description):
    content = f"""The following is a problem description:
{problem_description}
I am having trouble understanding the following code for the problem. Can you please list the relevant code smells in this code given the following problem description:
{code}"""
    messages = [
        {
            "content": content,
            "role": "user",
        }
    ]
    return await call_gpt_directly(messages)


async def get_refactoring_steps(prev_messages):
    prompt = """Great, can you refactor this code step-by-step (applying one refactoring at a time) to make it more understandable? Start from the previous version of the code, then output a new version."""
    messages = prev_messages + [
        {
            "content": prompt,
            "role": "user",
        }
    ]
    return await call_gpt_directly(messages)


async def get_final_refactored_code(prev_messages):
    prompt = """Thanks! Can you output the final version of the code. (Nothing else, no backticks or comment or anything like that.)"""
    messages = prev_messages + [
        {
            "content": prompt,
            "role": "user",
        }
    ]
    code_message = await call_gpt_directly(messages)
    if code_message is not None:
        code_message["content"] = parse_code(code_message["content"])
    return code_message


async def get_refactored_code_comparison(
    original_code, first_code, second_code, problem_description
):
    prompt = f"""I had two engineers refactor some code. Here's the original code
{original_code}

Here's a description of the problem the code is intended to solve
{problem_description} 

Refactoring 1:
{first_code}

Refactoring 2:
{second_code}

I want you to evaluate the refactoring from the two engineers. List the pros and cons of each refactoring, then state which refactoring is easier to understand and maintain."""
    messages = [
        {
            "content": prompt,
            "role": "user",
        }
    ]
    return await call_gpt_directly(messages)


async def refactor_code(path, code, problem_path) -> Dict[str, str]:
    """
    Get refactored code for the given problem.

    :param code: The code to refactor.
    :param problem_path: The path to the problem.
    :param max_tries: The maximum number of invalid refactored code to generate before raising an exception. Defaults to 4.
    """
    problem_question = get_question(problem_path)
    get_path = lambda x: os.path.join(path, x)
    os.makedirs(get_path(""), exist_ok=True)

    if not validate(code, problem_path):
        return {"end_reason": "original-invalid"}

    code_smells = await cache_wrapper(
        get_path("code_smells.txt"), get_code_smells, code, problem_question
    )
    if code_smells is None:
        return {"end_reason": "code smells prompt failed"}

    messages = [code_smells]
    refactoring_steps = await cache_wrapper(
        get_path("refactoring_steps.txt"), get_refactoring_steps, messages
    )
    if refactoring_steps is None:
        return {"end_reason": "refactoring steps prompt failed"}

    messages.append(refactoring_steps)
    final_refactored_code = await cache_wrapper(
        get_path("final_refactored_code.txt"), get_final_refactored_code, messages
    )
    if final_refactored_code is None:
        return {"end_reason": "final refactored code prompt failed"}

    # TODO: validate code properly
    final_refactored_code = final_refactored_code["content"]
    # if not validate(final_refactored_code, problem_path):
    #     return {"end_reason": "final refactored code invalid"}

    return {"end_reason": "success", "code": final_refactored_code}


async def get_best_refactoring(refactorings: List[str], problem_path: str) -> str:
    refactorings = []
    # TODO:
    # sort refactorings
    return refactorings[0]


async def generate_refactorings(
    problems, training_path, output_dir, attempts, solution_limit
):
    bar = tqdm(total=len(problems) * attempts)

    async def task(output_path, solution, problem_path):
        res = []
        for i in range(attempts):
            path = os.path.join(output_path, str(i))
            refactoring_res = await refactor_code(path, solution, problem_path)
            res.append(refactoring_res)
            bar.update(1)

        with open(os.path.join(output_path, "results.json"), "w") as f:
            json.dump(res, f, indent=4)

        successful_refactoring = [
            r["code"] for r in res if r["end_reason"] == "success"
        ]

        # get the best refactoring
        best_refactoring = get_best_refactoring(successful_refactoring, problem_path)

    async def tasks(problem):
        problem_path = os.path.join(training_path, problem)
        if not os.path.isdir(problem_path):
            return

        with open(os.path.join(problem_path, "solutions.json"), "r") as f:
            solutions = json.load(f)

        id = problem.split("/")[-1]
        res = []
        for i, solution in enumerate(solutions[:solution_limit]):
            path = os.path.join(output_dir, id, str(i))
            os.makedirs(path, exist_ok=True)
            res.append(task(path, solution, problem_path))

        results = await asyncio.gather(*res)
        with open(os.path.join(output_dir, id, "results.json"), "w") as f:
            json.dump(results, f, indent=4)

    await asyncio.gather(*[tasks(problem) for problem in problems])
    bar.close()


async def main(
    output_dir: str,
    start: int = 0,
    end: int = float("inf"),
    attempts: int = 1,
    solution_limit: int = 1,
):
    training_path = "APPS/train"
    problems = sorted(os.listdir(training_path))
    end = min(end, len(problems))
    problems = problems[start:end]
    await generate_refactorings(
        output_dir, training_path, problems, attempts, solution_limit
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir",
        type=str,
        help="The directory to output the refactored code to.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="The index of the first problem to refactor.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=float("inf"),
        help="The index of the last problem to refactor.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="The number of attempts to make for each problem.",
    )
    parser.add_argument(
        "--solution_limit",
        type=int,
        default=1,
        help="The number of solutions to refactor for each problem.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            args.output_dir,
            start=args.start,
            end=args.end,
            attempts=args.attempts,
            solution_limit=args.solution_limit,
        )
    )
