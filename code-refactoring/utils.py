from asyncio import sleep
from collections import defaultdict
from typing import Dict, List, Optional

import openai
import tiktoken
from aiolimiter import AsyncLimiter

TOKEN_LIMIT = 90000 / 3

chat_rate_limiter_gpt35 = AsyncLimiter(3500 / 1.5)
chat_token_limiter_gpt35 = AsyncLimiter(TOKEN_LIMIT)

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")


async def call_gpt(prompt: str) -> str:
    print("Calling GPT.")
    tokens = len(
        enc.encode(
            prompt
            + "You only output code. You do not output any commentary or anything else besides the actual code. Do not add any backticks or anything else to the code."
        )
    )
    await chat_rate_limiter_gpt35.acquire()
    if tokens > TOKEN_LIMIT:
        print("Skipping prompt due to token limit.")
        return None, False
    await chat_token_limiter_gpt35.acquire(tokens)
    for _ in range(10):
        try:
            messages = [
                {
                    "content": "You only output code. You do not output any commentary or anything else besides the actual code. Do not add any backticks or anything else to the code.",
                    "role": "system",
                },
                {
                    "content": prompt,
                    "role": "user",
                },
            ]
            completion = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=messages,
            )
            return completion.choices[0].message["content"], True
        except Exception as e:
            print(f"GPT Error: {e}")
            await sleep(10)
    return None, False


async def call_gpt_directly(messages: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    print("Calling GPT directly.")
    print(messages)
    tokens = len(enc.encode("".join([message["content"] for message in messages])))
    if tokens > TOKEN_LIMIT:
        print("Skipping prompt due to token limit.")
        return None

    await chat_token_limiter_gpt35.acquire(tokens)
    await chat_rate_limiter_gpt35.acquire()
    try:
        completion = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        return completion.choices[0].message
    except Exception as e:
        print(f"GPT Error: {e}")
    return None


def sort_python_code(code: List[str]) -> List[str]:
    """
    Sorts the methods such that a method is always called before it is defined.
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
