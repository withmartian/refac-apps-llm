import argparse
import asyncio
import json
import os
import random
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from tqdm.asyncio import tqdm

load_dotenv()


from utils.apps import get_problem_question, validate_code
from utils.gpt import Message, call_gpt, call_gpt_directly, parse_code_from_gpt
from utils.misc import cache_wrapper, get_json_with_default


async def get_gpt_conv(
    prev_messages: List[Message], content: str, model: str
) -> List[Message]:
    """
    Gets the messages from GPT conversation.

    :param prev_messages: The previous messages.
    :param content: The content of the message.
    :return: The messages from GPT conversation. (None if GPT fails)
    """

    messages = prev_messages + [{"content": content, "role": "user"}]
    message = await call_gpt_directly(messages, model)
    if message is None:
        return []
    messages.append(message)
    return messages


async def get_code_smells(
    code: str, problem_description: str, model: str
) -> List[Message]:
    """
    Requests the code smells for the given code and problem description.

    :param code: The code to get the code smells for.
    :param problem_description: The problem description to use.
    :param model: The model to use.
    :return: The messages from GPT conversation.
    """

    content = f"""
The following is a problem description:
```plaintext
{problem_description}
```
I am having trouble understanding the following code for the problem. Can you please list the relevant code smells in this code:
```python
{code}
```
"""
    return await get_gpt_conv([], content, model)


async def get_refactoring_steps(
    prev_messages: List[Message], model: str
) -> List[Message]:
    """
    Requests the refactoring steps from the given messages.

    :param prev_messages: The previous messages.
    :param model: The model to use.
    :return: The messages from GPT conversation.
    """

    content = """Great, can you refactor this code step-by-step (applying one refactoring at a time) to make it more understandable? Start from the previous version of the code, then output a new version."""
    return await get_gpt_conv(prev_messages, content, model)


async def get_final_refactored_code(
    prev_messages: List[Message], model: str
) -> List[Message]:
    """
    Requests the final refactored code from the given messages.

    :param prev_messages: The previous messages.
    :param model: The model to use.
    :return: The messages from GPT conversation.
    """

    content = """Thanks! Can you output the final version of the code. (Nothing else, no backticks or comment or anything like that.)"""
    return await get_gpt_conv(prev_messages, content, model)


async def get_refactored_code_comparison(
    original_code: str, codes: List[str], problem_description: str, model: str
) -> List[Message]:
    """
    Requests the refactored code comparison from the given messages.

    :param original_code: The original code.
    :param codes: The refactored codes.
    :param problem_description: The problem description to use.
    :param model: The model to use.
    :return: The messages from GPT conversation.
    """

    get_refactor = lambda i, code: f"Refactoring {i+1}:\n```python\n{code}\n```\n"
    n = len(codes)

    prompt = f"""I had {n} engineers refactor some code. Here's the original code:
```python
{original_code}
```

Here's a description of the problem the code is intended to solve:
```plaintext
{problem_description} 
```

Here are the refactorings from the {n} engineers:

{''.join(get_refactor(i, code) for i, code in enumerate(codes))}
I want you to evaluate the refactoring from the {n} engineers. List the pros and cons of each refactoring, then state which refactoring is easier to understand and maintain. When stating which is better, at the very end, output a number from 1 to {n} for the refactoring you think is better. Only output the number at the end, nothing else."""

    return await call_gpt(prompt, model) or []


async def refactor_code(
    path: str, code: str, problem_question: str, problem_path: str, model: str
) -> Dict[str, str]:
    """
    Refactors the given code.

    :param path: The path to save the code to or cache the steps in.
    :param code: The code to refactor.
    :param problem_question: The problem question to use.
    :param problem_path: The path to the problem.
    :param model: The model to use.
    :return: The end reason.
    """

    # create the path if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # helper function to get the path of a file
    get_path = lambda x: os.path.join(path, x)

    if not validate_code(code, problem_path):
        return {"end_reason": "original-invalid"}

    messages = await cache_wrapper(
        get_path("code_smells.json"),
        get_code_smells,
        code,
        problem_question,
        model,
    )
    if not messages:
        return {"end_reason": "code smells prompt failed"}

    messages = await cache_wrapper(
        get_path("refactoring_steps.json"),
        get_refactoring_steps,
        messages,
        model,
    )
    if not messages:
        return {"end_reason": "refactoring steps prompt failed"}

    messages = await cache_wrapper(
        get_path("final_refactored_code.json"),
        get_final_refactored_code,
        messages,
        model,
    )
    if not messages:
        return {"end_reason": "final refactored code prompt failed"}

    code = parse_code_from_gpt(messages[-1]["content"])
    if not validate_code(code, problem_path):
        return {"end_reason": "final refactored code invalid"}

    return {"end_reason": "success", "code": code}


def get_comparison_winner(comparison: str) -> Optional[int]:
    """
    Gets the winner of the comparison. Does this by scanning from the end of the comparison and finding the first number.
    (Note that this limits winner numbers to 0-9.)

    :param comparison: The comparison to get the winner of.
    :return: The winner of the comparison.
    """

    for i in range(len(comparison) - 1, -1, -1):
        if comparison[i].isdigit():
            return int(comparison[i])
    return None


async def get_best_pairwise_refactor(
    original_code: str,
    problem_description: str,
    refactors: List[str],
    save_path: str,
    model: str,
) -> Optional[str]:
    """
    Gets the best refactor for the given code using pairwise comparison.

    :param original_code: The original code.
    :param problem_description: The problem description to use.
    :param refactors: The refactors to use.
    :param save_path: The path to save the history to.
    :param attempts: The number of times to compare refactors.
    :param model: The model to use.
    :return: The best refactor. (None if comparison failed.)
    """

    def get_historical_best(history: List[Dict[str, Any]]) -> Optional[str]:
        if len(history) > 0:
            return (
                history[-1]["attacker"]
                if history[-1]["attacker_wins"]
                else history[-1]["defender"]
            )
        return None

    history_path = os.path.join(save_path, "history.json")
    history_obj = get_json_with_default(history_path)
    history = history_obj.get(get_best_pairwise_refactor.__name__, [])

    # make sure the history is valid
    assert len(history) == 0 or history[0]["defender"] == original_code

    # skip refactors that have already been tried
    attackers = set(x["attacker"] for x in history)
    rem_refactors = [refactor for refactor in refactors if refactor not in attackers]

    # shuffle the refactors in case the order matters
    order = list(range(len(rem_refactors)))
    random.shuffle(order)

    best_refactor = get_historical_best(history) or original_code
    for i in order:
        new_refactor = rem_refactors[i]

        # get the comparison
        comparison = await get_refactored_code_comparison(
            original_code, [best_refactor, new_refactor], problem_description, model
        )

        # get the winner of the comparison
        comparison_winner = get_comparison_winner(comparison)
        if comparison_winner not in [1, 2]:
            return None

        history.append(
            {
                "defender": best_refactor,
                "attacker": {
                    "code": new_refactor,
                    "original_order": i,
                },
                "attacker_wins": comparison_winner == 2,
                "fight": comparison,
            }
        )
        if history[-1]["attacker_wins"]:
            best_refactor = new_refactor

    # save the history
    with open(history_path, "w") as f:
        json.dump(history_obj, f, indent=4)
    return best_refactor


async def get_best_multinomial_refactor(
    original_code: str,
    problem_description: str,
    refactors: List[str],
    save_path: str,
    model: str,
) -> Optional[str]:
    """
    Gets the best refactor for the given code using a multinomial tournament.

    :param original_code: The original code.
    :param problem_description: The problem description to use.
    :param refactors: The refactors to use.
    :param save_path: The path to save the history to.
    :param model: The model to use.
    :return: The best refactor. (None if comparison failed.)
    """

    history_path = os.path.join(save_path, "history.json")
    history_obj = get_json_with_default(history_path)
    history = history_obj.get(get_best_multinomial_refactor.__name__, dict())

    rem_refactors = [r for r in refactors if r not in history.get("fighters", [])]
    best_refactor = history.get("winner") or original_code

    # if there are no refactors left, return the best refactor
    if len(rem_refactors) == 0:
        return best_refactor

    fighters = [best_refactor] + rem_refactors
    order = list(range(len(fighters)))
    random.shuffle(order)

    # get the comparison
    comparison = await get_refactored_code_comparison(
        original_code,
        [fighters[order[i]] for i in range(len(fighters))],
        problem_description,
        model,
    )

    # get the winner of the comparison
    comparison_winner = get_comparison_winner(comparison)
    if comparison_winner not in list(range(1, len(fighters) + 1)):
        return None

    # update the history
    history["comparisons"] = history.get("comparisons", []) + (
        [{"fighters": fighters, "comparison": comparison}]
    )
    history["original"] = original_code
    history["fighters"] = history.get("fighters", [])
    index = fighters[order[comparison_winner - 1]]
    if index != 0:
        # get index of the winner based on overall list of fighters
        history["winner"] = {
            "code": fighters[index],
            "original_order": index + len(history["fighters"]) - 1,
        }
    history["fighters"] += rem_refactors

    # save the history
    with open(history_path, "w") as f:
        json.dump(history_obj, f, indent=4)

    return history["winner"]


COMPARERS = [get_best_pairwise_refactor, get_best_multinomial_refactor]


def sort_python_code(code: List[str]) -> List[str]:
    """
    Sorts the methods such that a method is always called before it is defined.

    :param code: The code to sort.
    :return: The sorted code.
    """

    lines = code.split("\n")
    methods: List[List[str]] = [[]]
    for line in lines:
        if len(line) > 0 and line[0] not in [" ", "\t"]:
            if line.startswith("def"):
                methods.append([line])
            else:
                methods[0].append(line)
        else:
            methods[-1].append(line)

    res = topo_sort(methods[1:]) + [methods[0]]
    return "\n".join("\n".join(method) for method in res)


def topo_sort(methods: List[List[str]]) -> List[List[str]]:
    """
    Sorts the methods such that a method is always called before it is defined.

    :param methods: The methods to sort.
    :return: The sorted methods.
    """

    edges = defaultdict(list)
    names = [method[0].split(" ")[1].split("(")[0] for method in methods]
    for i, name in enumerate(names):
        for j, method in enumerate(methods):
            if i == j:
                continue

            if any(name in line for line in method):
                edges[i].append(j)

    res = []
    visited = set()

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for child in edges[node]:
            dfs(child)
        res.append(node)

    for i in range(len(methods)):
        dfs(i)

    return [methods[i] for i in res]


async def generate_refactoring(
    output_path: str,
    code: str,
    problem_path: str,
    comparers: List[Callable],
    num_refactors: int,
    pool_size: int,
    model: str,
) -> str:
    """
    Generates refactorings for a single solution.

    :param output_path: The path to save the refactorings to.
    :param code: The code to refactor.
    :param problem_path: The path to the problem.
    :param comparers: The comparers to use.
    :param num_refactors: The number of refactors to generate.
    :param pool_size: The number of refactors to generate in parallel.
    :param model: The model to use.
    :return: The final refactor.
    """

    problem_question = get_problem_question(problem_path)
    for i in range(num_refactors):
        path = os.path.join(output_path, "final_refactoring.txt")

        # check cache
        if os.path.exists(path):
            with open(path, "r") as f:
                code = f.read()
            continue

        # otherwise generate potential refactors
        refactor_path = os.path.join(output_path, f"refactor-{i}")
        os.makedirs(refactor_path, exist_ok=True)
        refactor_statuses = [
            refactor_code(
                os.path.join(refactor_path, f"attempt-{j}"),
                code,
                problem_question,
                problem_path,
                model,
            )
            for j in range(pool_size)
        ]
        refactor_statuses = await asyncio.gather(*refactor_statuses)

        # dump the refactor statuses
        with open(os.path.join(refactor_path, "status.json"), "w") as f:
            json.dump(refactor_statuses, f, indent=4)

        # filter for refactors that are valid
        refactors = [
            status["code"]
            for status in refactor_statuses
            if status["end_reason"] == "success"
        ]

        # if there are no refactors, break
        if len(refactors) == 0:
            print(f"Could not generate any refactors for {path}")
            break

        # get the best refactor
        c = Counter()
        for comparer in comparers:
            result = await comparer(
                code, problem_question, refactors, refactor_path, model
            )
            if result is not None:
                c[result] += 1

        # if there is no best refactor, break
        if len(c) == 0:
            print(f"Could not find best refactor for {path}")
            break

        # get the best refactor
        best_refactor = c.most_common(1)[0][0]

        # if the best refactor is the original code, break
        if best_refactor == code:
            break

        # save the best refactor
        with open(path, "w") as f:
            f.write(best_refactor)

        # update the code
        code = best_refactor

    return code


async def refactorings_main(
    problems: List[str],
    training_path: str,
    output_dir: str,
    num_refactors: int,
    pool_size: int,
    solution_limit: int,
    comparers: List[Callable],
    model: str,
) -> None:
    """
    Generates refactorings for each problem.

    :param problems: The problems to generate refactorings for.
    :param training_path: The path to the training data.
    :param output_dir: The directory to save the refactorings to.
    :param num_refactors: The number of refactors to generate.
    :param pool_size: The number of refactors to generate in parallel.
    :param solution_limit: The maximum number of solutions to generate refactorings for.
    :param comparers: The comparers to use to select the best refactor.
    :param model: The model to use.
    """

    async def task(problem: str):
        """
        Breaks down the problem into minitasks.

        :param problem: The problem to generate refactorings for.
        """

        minitasks = []
        problem_path = os.path.join(training_path, problem)
        with open(os.path.join(problem_path, "solutions.json"), "r") as f:
            solutions = json.load(f)

        id = problem.split("/")[-1]
        for i, solution in enumerate(solutions[:solution_limit]):
            path = os.path.join(output_dir, id, f"solution-{i}")
            minitasks.append(
                generate_refactoring(
                    path,
                    solution,
                    problem_path,
                    comparers,
                    num_refactors,
                    pool_size,
                    model,
                )
            )

        results = await tqdm.gather(*minitasks, desc=f"Refactorings for Problem {id}")
        with open(os.path.join(output_dir, id, "results.json"), "w") as f:
            json.dump(results, f, indent=4)

    await tqdm.gather(*[task(problem) for problem in problems], desc=f"Problems")


async def main(
    start: int,
    end: int,
    output_dir: str,
    training_path: str,
    num_refactors: int,
    pool_size: int,
    solution_limit: int,
    comparers: List[int],
    model: str,
) -> None:
    """
    Main function.

    :param output_dir: The directory to save the refactorings to.
    :param training_path: The path to the training data.
    :param start: The index of the first problem to refactor.
    :param end: The index of the last problem to refactor.
    :param num_refactors: The number of refactors to generate.
    :param pool_size: The number of refactors to generate in parallel.
    :param solution_limit: The maximum number of solutions to generate refactorings for.
    :param comparers: The comparers to use to select the best refactor.
    :param model: The model to use.
    """

    problems = sorted(os.listdir(training_path))
    start = max(start, 0)
    end = min(end, len(problems))
    problems = problems[start:end]
    comparers = [COMPARERS[i] for i in comparers]
    await refactorings_main(
        problems,
        training_path,
        output_dir,
        num_refactors,
        pool_size,
        solution_limit,
        comparers,
        model,
    )


def parse_args():
    parser = argparse.ArgumentParser()
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
        "--output-dir",
        type=str,
        default="refactor-results",
        help="The directory to output the refactored code to.",
    )
    parser.add_argument(
        "--training-path",
        type=str,
        default="APPS/train",
        help="The path to the training problems.",
    )
    parser.add_argument(
        "--num-refactors",
        type=int,
        default=1,
        help="The number of refactors to generate.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=1,
        help="The number of refactors to generate in parallel.",
    )
    parser.add_argument(
        "--solution-limit",
        type=int,
        default=1,
        help="The number of solutions to refactor for each problem.",
    )
    parser.add_argument(
        "--comparers",
        nargs="+",
        type=int,
        choices=list(range(len(COMPARERS))),
        default=[1],
        help="The indices of the comparers to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="The model to use.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(**vars(args)))
