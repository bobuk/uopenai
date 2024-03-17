import httpx
from typing import Dict, Any, Optional, List, Type, TypeVar
from datetime import datetime
from pydantic import BaseModel
from copy import deepcopy
from json import loads, dumps
OAI_V1 = "https://api.openai.com/v1/"

T = TypeVar('T', bound=BaseModel)

class Message(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    index: int
    message: Message
    logprobs: Optional[Any]
    finish_reason: Optional[str]

    @property
    def json(self) -> Optional[dict]:
        if not self.message:
            return None
        try:
            return loads(self.message.content)
        except:
            return None



class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class Answer(BaseModel):
    id: str
    object: str
    choices: List[Choice]
    created: datetime
    model: str
    system_fingerprint: str
    usage: Usage


class OpenAIError(Exception):
    pass

class OpenAI:
    def __init__(self, api_key: str, organization: Optional[str] = None):
        self._api_key = api_key
        self._auth_header = deepcopy({"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        if organization:
            self._auth_header["OpenAI-Organization"] = organization

    async def call(self, endpoint: str, return_type: Type[T], payload: Dict={}) -> T:
        url = OAI_V1 + endpoint
        headers = self._auth_header
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                raise OpenAIError(f"Error: {response.status_code} - {response.text}")
            js = response.json()
            return return_type.model_validate(js)

    async def complete(self, messages: List[Message],
        model: str = "gpt-3.5-turbo",
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        json: Optional[bool] = None,
        temperature: Optional[float] = None,
        **payload) -> Answer:
        req = {}



        params = locals()
        for opt in ["model", "max_tokens", "presence_penalty", "frequency_penalty", "temperature"]:
            if opt in params and params[opt] is not None:
                req[opt] = params[opt]

        if json:
            req["response_format"] = {"type":"json_object"}

        req_messages = []
        for msg in messages:
            req_messages.append(msg.model_dump())

        req["messages"] = req_messages
        response = await self.call("chat/completions", Answer, req)
        return response

    async def easy_complete(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> dict | str:
        if system_prompt:
            messages = [Message(role="system", content=system_prompt), Message(role="user", content=prompt)]
        else:
            messages = [Message(role="user", content=prompt)]
        answer = await self.complete(messages, **kwargs)
        if not answer.choices:
            raise OpenAIError("No response from OpenAI")
        if 'json' in kwargs and kwargs['json']:
            dict_result = answer.choices[0].json
            if not dict_result:
                raise OpenAIError("No JSON response from OpenAI")
            return dict_result
        return answer.choices[0].message.content
