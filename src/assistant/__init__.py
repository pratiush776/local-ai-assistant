import json

import httpx

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:9b"
SYSTEM_PROMPT = "You are an helpful assistant whose main job is to serve and help user however you can."
TEMPERATURE = 0.7
CONTEXT_BUDGET = 300


def ask_without_stream(messages: list[dict]) -> str:
    payload = {"model": MODEL, "messages": messages, "think": False, "stream": False}
    response = httpx.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


def ask_with_stream(messages: list[dict]) -> tuple[str, int]:
    payload = {
        "model": MODEL,
        "messages": messages,
        "think": False,
        "stream": True,
        "options": {
            "temperature": TEMPERATURE,
        },
    }
    reply = ""
    context = 0
    with httpx.stream("POST", OLLAMA_URL, json=payload, timeout=120) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk["done"]:
                output_tokens = chunk["eval_count"]
                input_tokens = chunk["prompt_eval_count"]
                output_duration = (
                    chunk["eval_duration"] / 1000000000
                )  # convert to seconds
                ttft = (
                    chunk["load_duration"] + chunk["prompt_eval_duration"]
                ) / 1000000000  # time to first token

                context = input_tokens + output_tokens
                tokens_per_sec = (output_tokens) / (output_duration)

                print(
                    f"\n[ Context: {context} tokens ]",
                    f"\n[ {output_tokens} tokens out -> {tokens_per_sec:.1f} tok/s . first token {ttft:.1f}s ]\n"
                )
            piece = chunk["message"]["content"]
            print(piece, end="", flush=True)
            reply += piece
    return reply, context


def main() -> None:
    context = 0
    messages = []
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
    user_msg = input("Enter your message: ")
    while user_msg.lower() != "quit":
        messages.append({"role": "user", "content": user_msg})

        # est. context check before inference
        estimated_prompt_context = len(user_msg) / 4
        removed_count = 0
        while len(messages) > 2 and context + estimated_prompt_context > CONTEXT_BUDGET:
            for i in range(2): #remove pairs of users + assistant message so that message doesn't start with assistant response.
                removed_msg = messages.pop(1)
                context -= len(removed_msg["content"]) / 4
                removed_count += 1

        if removed_count != 0:
            print(f"[Trimmed {removed_count} messages]")

        # api call to LLM
        try:
            assistant_msg, response_context = ask_with_stream(list(messages))
            context = response_context
            messages.append({"role": "assistant", "content": assistant_msg})
        except httpx.HTTPError as e:  # error handling
            print("Error: ", e)
            messages.pop()

        user_msg = input("Enter your message: ")
