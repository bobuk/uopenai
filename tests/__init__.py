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
async def test_easy_complete_with_valid_prompts():
    openai = OpenAI(api_key=get_key())
    prompt = "Tell me a fun fact about the ocean."
    system_prompt = "You are a knowledgeable bot."

    response = await openai.easy_complete(prompt, system_prompt=system_prompt, model="gpt-4-turbo-preview")

    assert isinstance(response, Answer)
    assert response.choices[0].message.content is not None
    assert "ocean" in response.choices[0].message.content

@pytest.mark.asyncio
async def test_easy_complete_without_system_prompt():
    openai = OpenAI(api_key=get_key())
    prompt = "What's the weather like today?"

    response = await openai.easy_complete(prompt, model="gpt-4-turbo-preview")

    assert isinstance(response, Answer)
    assert response.choices[0].message.content is not None

@pytest.mark.asyncio
async def test_easy_complete_with_empty_prompt():
    openai = OpenAI(api_key=get_key())

    # must work
    response = await openai.easy_complete("", model="gpt-4-turbo-preview")

@pytest.mark.asyncio
async def test_easy_complete_with_invalid_model():
    openai = OpenAI(api_key=get_key())
    prompt = "What is the capital of France?"

    with pytest.raises(OpenAIError):
        response = await openai.easy_complete(prompt, model="invalid-model")

@pytest.mark.asyncio
async def test_easy_complete_with_additional_kwargs():
    openai = OpenAI(api_key=get_key())
    prompt = "Write a poem about a sunset."
    system_prompt = "You are a creative bot."
    kwargs = {'max_tokens': 50, 'temperature': 0.7}

    response = await openai.easy_complete(prompt, system_prompt=system_prompt, **kwargs, model="gpt-4-turbo-preview")

    assert isinstance(response, Answer)
    assert response.choices[0].message.content is not None
    assert len(response.choices[0].message.content.split()) <= 50

@pytest.mark.asyncio
async def test_easy_complete_translate():
    openai = OpenAI(api_key=get_key())
    prompt = "Translate 'Hello, how are you?' into Spanish."
    system_prompt = "You are a multilingual bot."

    response = await openai.easy_complete(prompt, system_prompt=system_prompt, model="gpt-4-turbo-preview")

    assert isinstance(response, Answer)
    assert response.choices[0].message.content is not None
    assert "Hola" in response.choices[0].message.content

if __name__ == '__main__':
    import httpx
    # import logging
    # logging.basicConfig(
    #     format="%(levelname)s [%(asctime)s] %(name)s - %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    #     level=logging.DEBUG
    # )

    asyncio.run(test_complete())
