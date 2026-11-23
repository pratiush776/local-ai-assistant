import time

import anthropic

client = anthropic.Anthropic()  # reads api key from env


def claude_chat(
    messages: list,
    system: str,
    model: str,
    max_tokens: int = 1024,
) -> tuple[str, dict]:
    reply = ""

    start_time = time.perf_counter()
    first_token_time = None
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        system=system,
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            if not first_token_time:
                first_token_time = time.perf_counter()
            reply += text
        final = stream.get_final_message()
    end_time = time.perf_counter()
    # stats build
    output_tokens = final.usage.output_tokens
    input_tokens = final.usage.input_tokens

    tokens_per_sec = (
        round(output_tokens / (end_time - first_token_time), 1)
        if first_token_time
        else None
    )
    context = output_tokens + input_tokens
    ttft = round(first_token_time - start_time, 1) if first_token_time else None

    return reply, {
        "context": context,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_sec": tokens_per_sec,
        "ttft": ttft,
    }
