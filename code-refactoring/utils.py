from asyncio import sleep

import openai
import tiktoken
from aiolimiter import AsyncLimiter

TOKEN_LIMIT = 90000 / 3

chat_rate_limiter_gpt35 = AsyncLimiter(3500 / 1.5)
chat_token_limiter_gpt35 = AsyncLimiter(TOKEN_LIMIT)

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")


async def call_gpt(prompt: str) -> str:
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
