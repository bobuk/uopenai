import asyncio
import pytest

import unittest
import os
import httpx
from uopenai import OpenAI, Message, OpenAIError, Answer

def get_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not set")
    return key

@pytest.mark.asyncio
async def test_complete():
    openai = OpenAI(api_key=get_key())
    messages = [
        Message(role="system", content="You are a chatbot."),
        Message(role="user", content="What is your name?")
    ]

    response = await openai.complete(messages, model="gpt-4-turbo-preview")

    assert(response.choices[0].message.content is not None)
    with pytest.raises(Exception):
        response = await openai.complete(messages, model="gpt-nonexistent-model")

@pytest.mark.asyncio
async def test_complete_with_empty_message():
    openai = OpenAI(api_key=get_key())
    messages = []

    with pytest.raises(OpenAIError):
        response = await openai.complete(messages, model="gpt-4-turbo-preview")

@pytest.mark.asyncio
async def test_complete_with_invalid_api_key():
    openai = OpenAI(api_key="invalidkey")
    messages = [
        Message(role="system", content="You are a chatbot."),
        Message(role="user", content="Hello, who are you?")
    ]

    with pytest.raises(OpenAIError):
        response = await openai.complete(messages, model="gpt-4-turbo-preview")


@pytest.mark.asyncio
async def test_complete_with_multiple_messages():
    openai = OpenAI(api_key=get_key())
    messages = [
        Message(role="system", content="You are a chatbot."),
        Message(role="user", content="Hello!"),
        Message(role="user", content="Can you tell me a joke?"),
        Message(role="system", content="Sure, why don't scientists trust atoms?"),
        Message(role="system", content="Because they make up everything!")
    ]

    response = await openai.complete(messages, model="gpt-4-turbo-preview")

    assert len(response.choices) > 0
    assert response.choices[0].message.content is not None

@pytest.mark.asyncio
async def test_complete_with_json_response_format():
    openai = OpenAI(api_key=get_key())
    messages = [
        Message(role="system", content="You are a chatbot."),
        Message(role="user", content="Can you provide a JSON formatted response?")
    ]

    response = await openai.complete(messages, model="gpt-4-turbo-preview", json=True)
    print(response.choices[0].json)

@pytest.mark.asyncio
async def test_easy_complete_with_json():
    openai = OpenAI(api_key=get_key())
    response = await openai.easy_complete("what is the capital of France? answer in JSON, put the name in `capital`", json=True)
    assert isinstance(response, dict)
    assert "capital" in response
    assert response["capital"] == "Paris"

if __name__ == '__main__':
    import httpx
    # import logging
    # logging.basicConfig(
    #     format="%(levelname)s [%(asctime)s] %(name)s - %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    #     level=logging.DEBUG
    # )

    asyncio.run(test_complete_with_json_response_format())
    # asyncio.run(test_complete())
