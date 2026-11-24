import json

import httpx

OLLAMA_URL = "http://localhost:11434/api/chat"
LOCAL_CONTEXT_WINDOW = 8192
TEMPERATURE = 0.7


def chat_message(
    messages: list[dict],
    model: str,
    *,
    tools: list[dict] | None = None,
    options: dict[str, int | float] | None = None,
    format_schema: dict | None = None,
) -> dict:
    """Return Ollama's assistant message, including any proposed tool calls."""
    payload = {
        "model": model,
        "messages": messages,
        "think": False,
        "stream": False,
        "options": {"num_ctx": LOCAL_CONTEXT_WINDOW, **(options or {})},
    }
    if tools is not None:
        payload["tools"] = tools
    if format_schema is not None:
        payload["format"] = format_schema
    response = httpx.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["message"]


def ask_without_stream(
    messages: list[dict],
    model: str,
    *,
    options: dict[str, int | float] | None = None,
    format_schema: dict | None = None,
) -> str:
    message = chat_message(
        messages, model, options=options, format_schema=format_schema
    )
    return message["content"]


def ask_with_stream(
    messages: list[dict], model: str, warmup: bool = False
) -> tuple[str, dict]:
    payload = {
        "model": model,
        "messages": messages,
        "think": False,
        "stream": True,
        "options": {
            "num_ctx": LOCAL_CONTEXT_WINDOW,
            "temperature": TEMPERATURE,
        },
    }
    reply = ""
    context, output_tokens, input_tokens, tokens_per_sec, ttft = (
        0,
        None,
        None,
        None,
        None,
    )
    with httpx.stream("POST", OLLAMA_URL, json=payload, timeout=120) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or warmup:
                continue
            chunk = json.loads(line)
            if chunk["done"]:
                output_tokens = chunk["eval_count"]
                input_tokens = chunk["prompt_eval_count"]
                output_duration = (
                    chunk["eval_duration"] / 1000000000
                )  # convert to seconds
                ttft = round(
                    (chunk["load_duration"] + chunk["prompt_eval_duration"])
                    / 1000000000,
                    1,
                )  # time to first token

                context = input_tokens + output_tokens
                tokens_per_sec = round((output_tokens) / (output_duration), 1)
            piece = chunk["message"]["content"]
            print(piece, end="", flush=True)
            reply += piece
    if warmup:
        print("local model warmed up")
    return reply, {
        "context": context,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_sec": tokens_per_sec,
        "ttft": ttft,
    }
