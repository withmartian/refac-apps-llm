from typing import Dict, List, Optional

import openai
import tiktoken
from aiolimiter import AsyncLimiter

############################################
########## GPT SPECIFIC METHODS ############
############################################


MODEL_SPECIFIC_METHODS = {
    "gpt-3.5-turbo": {
        "max_tokens": 90000,
        "rate_limiter": AsyncLimiter(3500),
        "token_limiter": AsyncLimiter(90000),
    },
    "gpt-4": {
        "max_tokens": 40000,
        "rate_limiter": AsyncLimiter(200),
        "token_limiter": AsyncLimiter(40000),
    },
}

Message = Dict[str, str]


async def passes_limiter_checks(model: str, messages: List[Message]) -> bool:
    """
    Routes the request to the correct limiter.

    :param model: The model to use.
    :return: Whether the request was successful (depending on the token limit)
    """

    if model not in MODEL_SPECIFIC_METHODS:
        raise ValueError(f"Unknown model: {model}")

    enc = tiktoken.encoding_for_model(model)
    content = "\n".join(
        f"{message['role']}: {message['content']}" for message in messages
    )
    token_size = len(enc.encode(content))

    if token_size > MODEL_SPECIFIC_METHODS[model]["max_tokens"]:
        return False

    await MODEL_SPECIFIC_METHODS[model]["rate_limiter"].acquire()
    await MODEL_SPECIFIC_METHODS[model]["token_limiter"].acquire(token_size)
    return True


async def call_gpt(prompt: str, model: str = "gpt-3.5-turbo") -> Optional[str]:
    """
    Calls GPT with the given prompt.

    :param prompt: The prompt to use.
    :param model: The model to use. (Optional, defaults to gpt-3.5-turbo)
    :return: The string response from GPT.
    """

    messages = [
        {
            "content": "You only output code. You do not output any commentary or anything else besides the actual code. Do not add any backticks or anything else to the code.",
            "role": "system",
        },
        {"content": prompt, "role": "user"},
    ]
    res = await call_gpt_directly(messages, model)
    if res is not None:
        return res["content"]
    return None


async def call_gpt_directly(
    messages: List[Message], model: str = "gpt-3.5-turbo"
) -> Optional[Message]:
    """
    Calls GPT with the given messages.

    :param messages: The messages to use.
    :param model: The model to use. (Optional, defaults to gpt-3.5-turbo)
    :return: The message response from GPT.
    """

    if not await passes_limiter_checks(model, messages):
        print("Skipping prompt due to token limit.")
        return None

    try:
        completion = await openai.ChatCompletion.acreate(model=model, messages=messages)
        return completion.choices[0].message
    except Exception as e:
        print(f"GPT Error: {e}")
    return None


def parse_code_from_gpt(text: str) -> str:
    """
    If the text is a code block, returns the code block.

    :param text: The text to parse.
    :return: The code block.
    """

    if "```" in text:
        text = text.split("```")[1]
    if text.startswith("python"):
        text = text[6:]
    return text
