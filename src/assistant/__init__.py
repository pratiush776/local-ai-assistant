import anthropic
import httpx

from .claude import claude_chat
from .context import estimate_tokens, trim_history
from .ollama import ask_with_stream

SYSTEM_PROMPT = (
    "You are an helpful assistant whose main job is to help users ethically."
)
CONTEXT_BUDGET = 4096
MODELS = ["qwen3.5:9b", "claude-haiku-5-5"]


def prompt_for_model_idx() -> int:
    # prompt user to choose a model
    for i, model in enumerate(MODELS):
        print(f"{i} {model}")
    while True:
        selected_model_idx = input(f"Select a model (0 - {len(MODELS) - 1}): ")
        try:
            idx = int(selected_model_idx)
        except ValueError:
            print("Please enter a valid number.")
            continue
        if 0 <= idx < len(MODELS):
            return idx

        print("Out of range.")


def select_model(model_idx: int) -> str:
    m = MODELS[model_idx]
    print("Model selected: ", m)
    return m


def prompt_select_model() -> str:
    i = prompt_for_model_idx()
    m = select_model(i)
    prewarm_cache(m)
    return m


def prewarm_cache(model: str) -> None:
    if model.startswith("claude"):
        return
    ask_with_stream([], model, warmup=True)


def ask(model: str, messages: list, system: str) -> tuple[str, dict]:
    if model.startswith("claude"):
        return claude_chat(messages=messages, model=model, system=system)
    return ask_with_stream(
        [{"role": "system", "content": system}] + messages, model=model
    )


def print_stats(stats: dict) -> None:
    print(
        f"\n[ context {stats['context']} toks . {stats['output_tokens']} out"
        f" . {stats['tokens_per_sec']} tok/s . first token {stats['ttft']}s ]"
    )


def chat():
    context = 0
    primary_model = prompt_select_model()
    context += estimate_tokens(SYSTEM_PROMPT)
    messages = []
    usr_msg = input("Enter Message: ")
    while usr_msg != "quit":
        if usr_msg.startswith("/"):
            command = "".join(usr_msg[1:].split())
            if command in MODELS:
                primary_model = command
                print(f"Model Selected: {primary_model}")
            else:
                print("Invalid Model")
                primary_model = prompt_select_model()
        else:
            try:
                messages.append({"role": "user", "content": usr_msg})
                # send conversation with new msg to model
                res, stats = ask(
                    model=primary_model, messages=messages, system=SYSTEM_PROMPT
                )
                print_stats(stats)
                messages.append({"role": "assistant", "content": res})
                context = stats["context"]
            except (
                httpx.HTTPError,
                anthropic.APIConnectionError,
                anthropic.APIStatusError,
            ):
                print("Network Error: Unable to connect to server!")
                msg = messages.pop()
                context -= estimate_tokens(msg["content"])
        usr_msg = input("Enter message: ")
        # CONTEXT MANAGEMENT with estimated context and prev context
        context += estimate_tokens(usr_msg)
        messages, context = trim_history(messages, context, CONTEXT_BUDGET)


def main() -> None:
    chat()
