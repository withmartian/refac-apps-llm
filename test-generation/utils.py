import openai


async def call_gpt(messages):
    completion = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    return completion.choices[0].message
