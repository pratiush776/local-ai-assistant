from typing import Literal

import httpx
from pydantic import BaseModel


class Request(BaseModel):
    text:str
    label: Literal["chat","fact", "reasoning","action"]

class Batch(BaseModel):
    requests: list[Request]

payload = {
    "model":"qwen3.5:9b-q4_K_M",
    "messages":[
        {
            "role":"user",
            "content":"Write 5 short things a person might type to their personal AI assistant."
            "Label each one: chat, fact, reasoning or action"
        }
    ],
    "format": Batch.model_json_schema(),
    "think":False,
    "stream":False
}

response = httpx.post('http://localhost:11434/api/chat',json=payload,timeout=300)
response.raise_for_status()
data = response.json()

batch = Batch.model_validate_json(data["message"]["content"])
for req in batch.requests:
    print(f"{req.label:10} {req.text}")

print(data["prompt_eval_count"],data["eval_count"])