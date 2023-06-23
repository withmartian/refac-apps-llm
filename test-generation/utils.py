from asyncio import sleep
from typing import Optional, Tuple

import openai
import tiktoken
from aiolimiter import AsyncLimiter

TOKEN_LIMIT = 90000 / 3

chat_rate_limiter_gpt35 = AsyncLimiter(3500 / 1.5)
chat_token_limiter_gpt35 = AsyncLimiter(TOKEN_LIMIT)

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")


async def call_gpt(prompt: str) -> Optional[str]:
    tokens = len(enc.encode(prompt))
    await chat_rate_limiter_gpt35.acquire()
    if tokens > TOKEN_LIMIT:
        print("Skipping prompt due to token limit.")
        return None
    await chat_token_limiter_gpt35.acquire(tokens)
    for _ in range(10):
        try:
            messages = [
                {
                    "content": prompt,
                    "role": "user",
                },
            ]
            completion = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=messages,
            )
            return completion.choices[0].message["content"]
        except Exception as e:
            print(f"Error: {e}")
            await sleep(10)
    return None
