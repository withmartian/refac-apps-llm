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
from utils import call_gpt_directly, call_gpt


def validate(code, problem_path) -> bool:
    os.makedirs("refactor-temp", exist_ok=True)
    if os.path.exists("refactor-temp/all_results.json"):
        os.remove("refactor-temp/all_results.json")
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
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with open("refactor-temp/all_results.json", "r") as f:
            body = json.load(f)
            print(f"body: {body}")
            # check if it is a list of booleans
            if not isinstance(body["0"][0][0], bool):
                print("not bool")
                return False
            res = np.all(body["0"])
            print(f"res: {res}")
            return res
    except:
        return False


def get_problem_question(problem_path):
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
            if path.endswith(".json"):
                return json.load(f)
            else:
                return f.read()
    else:
        res = await func(*args, **kwargs)
        if res is None:
            return None
        with open(path, "w") as f:
            if path.endswith(".json"):
                json.dump(res, f, indent=4)
            else:
                f.write(res)
        return res


async def safely_add_to_messages(messages, content) -> List[Dict[str, str]]:
    messages.append({"content": content, "role": "user"})
    if res := await call_gpt_directly(messages):
        messages.append(res)
        return messages
    else:
        messages.pop()
        return None


async def get_code_smells(code, problem_description, messages):
    assert len(messages) == 0
    content = f"""The following is a problem description:
{problem_description}
I am having trouble understanding the following code for the problem. Can you please list the relevant code smells in this code given the following problem description:
{code}"""
    return await safely_add_to_messages(messages, content)


async def get_refactoring_steps(messages):
    content = """Great, can you refactor this code step-by-step (applying one refactoring at a time) to make it more understandable? Start from the previous version of the code, then output a new version."""
    return await safely_add_to_messages(messages, content)


async def get_final_refactored_code(messages):
    content = """Thanks! Can you output the final version of the code. (Nothing else, no backticks or comment or anything like that.)"""
    return await safely_add_to_messages(messages, content)


async def get_refactored_code_comparison(
    original_code, codes: List[str], problem_description
):
    def get_refactor(i, code):
        return f"Refactoring {i+1}:\n{code}\n"

    n = len(codes)
    prompt = f"""I had {n} engineers refactor some code. Here's the original code:
{original_code}

Here's a description of the problem the code is intended to solve:
{problem_description} 

{''.join(get_refactor(i, code) for i, code in enumerate(codes))}
I want you to evaluate the refactoring from the {n} engineers. List the pros and cons of each refactoring, then state which refactoring is easier to understand and maintain. When stating which is better, at the very end, output a number from 1 to {n} for the refactoring you think is better."""
    return (await call_gpt(prompt))[0]


async def refactor_code(path, code, problem_question, problem_path) -> Dict[str, str]:
    get_path = lambda x: os.path.join(path, x)
    os.makedirs(path, exist_ok=True)

    if not validate(code, problem_path):
        return {"end_reason": "original-invalid"}

    messages = []
    messages = await cache_wrapper(
        get_path("code_smells.json"), get_code_smells, code, problem_question, messages
    )
    if messages is None:
        return {"end_reason": "code smells prompt failed"}

    messages = await cache_wrapper(
        get_path("refactoring_steps.json"), get_refactoring_steps, messages
    )
    if messages is None:
        return {"end_reason": "refactoring steps prompt failed"}

    messages = await cache_wrapper(
        get_path("final_refactored_code.json"), get_final_refactored_code, messages
    )
    if messages is None:
        return {"end_reason": "final refactored code prompt failed"}
    messages[-1]["content"] = parse_code(messages[-1]["content"])

    code = messages[-1]["content"]
    if not validate(code, problem_path):
        return {"end_reason": "final refactored code invalid"}

    return {"end_reason": "success", "code": code}


def get_existing_history(save_path):
    history_path = os.path.join(save_path, "history.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            try:
                return json.load(f)
            except:
                print("Failed to load history for problem", save_path)
                pass
    return []


def get_historical_best(history):
    if len(history) > 0:
        return (
            history[-1]["attacker"]
            if history[-1]["attacker_wins"]
            else history[-1]["defender"]
        )
    return None


async def get_best_refactor(
    original_code,
    problem_description,
    refactors: List[str],
    save_path: str,
) -> str:
    history = get_existing_history(save_path)
    assert len(history) == 0 or history[0]["defender"] == original_code

    # skip refactors that have already been tried
    rem_refactors = [
        refactor
        for refactor in refactors
        if refactor not in [x["attacker"] for x in history]
    ]

    best_refactor = get_historical_best(history) or original_code
    for new_refactor in rem_refactors:
        comparison = await get_refactored_code_comparison(
            original_code, [best_refactor, new_refactor], problem_description
        )
        comparison_winner = get_comparison_winner(comparison)
        if comparison_winner is None or comparison_winner not in [1, 2]:
            return None
        history.append(
            {
                "defender": best_refactor,
                "attacker": new_refactor,
                "fight": comparison,
                "attacker_wins": comparison_winner == 2,
            }
        )
        if history[-1]["attacker_wins"]:
            best_refactor = new_refactor

        # cache history
        with open(os.path.join(save_path, "history.json"), "w") as f:
            json.dump(history, f, indent=4)
    return best_refactor


def get_existing_history_v2(save_path):
    history_path = os.path.join(save_path, "history2.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            try:
                return json.load(f)
            except:
                print("Failed to load history for problem", save_path)
                pass
    return {
        "original": None,
        "fighters": [],
        "winner": None,
        "comparisons": [],
    }


def get_comparison_winner(comparison):
    for i in range(len(comparison) - 1, -1, -1):
        if comparison[i].isdigit():
            return int(comparison[i])
    return None


async def get_best_refactor_v2(
    original_code,
    problem_description,
    refactors: List[str],
    save_path: str,
) -> str:
    history = get_existing_history_v2(save_path)
    rem_refactors = [
        refactor for refactor in refactors if refactor not in history["fighters"]
    ]
    best_refactor = history["winner"] or original_code

    if len(rem_refactors) == 0:
        return best_refactor

    # get prompt for best refactoring among all
    fighters = [best_refactor] + rem_refactors
    comparison = await get_refactored_code_comparison(
        original_code, fighters, problem_description
    )
    comparison_winner = get_comparison_winner(comparison)
    if comparison_winner is None or comparison_winner not in list(
        range(1, len(fighters) + 1)
    ):
        return None

    history["comparisons"].append({"fighters": fighters, "comparison": comparison})
    history["original"] = original_code
    history["fighters"] += rem_refactors
    history["winner"] = fighters[comparison_winner - 1]

    with open(os.path.join(save_path, "history2.json"), "w") as f:
        json.dump(history, f, indent=4)

    return history["winner"]


COMPARERS = [get_best_refactor, get_best_refactor_v2]


async def generate_refactorings(
    problems, training_path, output_dir, attempts, solution_limit, comparers
):
    bar = tqdm(total=len(problems) * attempts)

    async def task(output_path, solution, problem_path):
        problem_question = get_problem_question(problem_path)
        get_path = lambda i: os.path.join(output_path, f"attempt-{i}")
        mini_tasks = [
            refactor_code(get_path(i), solution, problem_question, problem_path)
            for i in range(attempts)
        ]
        results = await asyncio.gather(*mini_tasks)

        # show results of all attempts to refactor
        with open(os.path.join(output_path, "results.json"), "w") as f:
            json.dump(results, f, indent=4)

        # filter out unsuccessful refactorings
        successful_refactors = [
            result["code"] for result in results if result["end_reason"] == "success"
        ]

        # get existing best refactorings
        res = dict()
        for comparer in comparers:
            best_refactor = await comparer(
                solution, problem_question, successful_refactors, output_path
            )
            res[comparer.__name__] = best_refactor

        with open(os.path.join(output_path, "best_refactors.json"), "w") as f:
            json.dump(res, f, indent=4)
        return res

    async def tasks(problem):
        problem_path = os.path.join(training_path, problem)
        if not os.path.isdir(problem_path):
            return

        with open(os.path.join(problem_path, "solutions.json"), "r") as f:
            solutions = json.load(f)

        id = problem.split("/")[-1]
        res = []
        for i, solution in enumerate(solutions[:solution_limit]):
            path = os.path.join(output_dir, id, f"solution-{i}")
            os.makedirs(path, exist_ok=True)
            res.append(task(path, solution, problem_path))

        results = await asyncio.gather(*res)
        bar.update(attempts)
        with open(os.path.join(output_dir, id, "results.json"), "w") as f:
            json.dump(results, f, indent=4)

    await asyncio.gather(*[tasks(problem) for problem in problems])
    bar.close()


async def main(
    output_dir: str,
    training_path: str,
    start: int,
    end: int,
    attempts: int,
    solution_limit: int,
    comparers: List[int],
):
    problems = sorted(os.listdir(training_path))
    start = max(start, 0)
    end = min(end, len(problems))
    problems = problems[start:end]
    comparers = [COMPARERS[i] for i in comparers]
    await generate_refactorings(
        problems, training_path, output_dir, attempts, solution_limit, comparers
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
        "--solution-limit",
        type=int,
        default=1,
        help="The number of solutions to refactor for each problem.",
    )
    parser.add_argument(
        "--training-path",
        type=str,
        default="APPS/train",
        help="The path to the training problems.",
    )
    parser.add_argument(
        "--comparers",
        nargs="+",
        type=int,
        choices=list(range(len(COMPARERS))),
        default=[1],
        help="The indices of the comparers to use.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(**vars(args)))
