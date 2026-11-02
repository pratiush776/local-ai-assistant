import anthropic
import httpx

from .claude import claude_chat
from .context import estimate_tokens, messages_to_text, trim_history
from .decide import Decision, decide
from .ollama import ask_with_stream, ask_without_stream

SYSTEM_PROMPT = (
    "You are an helpful assistant whose main job is to help users ethically."
)
CONTEXT_BUDGET = 4096
MODELS = ["qwen3.5:9b", "claude-haiku-5-5"]
AUTO_ESCALATE = False
PRIMARY_LOCAL_MODEL = MODELS[0]
ESCALATE_MODEL = MODELS[1]


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


def run_ctx_clarification(cur_msg_ctx: list) -> str:
    cur_msg_ctx_txt = messages_to_text(cur_msg_ctx)
    clarify_context_prompt = f"""Based on the conversation below, give out a final improved prompt in first person as if it was given by the user at once. It should include all the relevant information and the ask end to end as if it would be in a normal conversation.
    The conversation exchange from the unclear message from the user that you have to compact and make one to be appended in the main context history:
    {cur_msg_ctx_txt}
    Note: if there is jsut one exchange that is user then just make it compact without losing the meaning."""
    clarified_context = ask_without_stream(
        [{"role": "user", "content": clarify_context_prompt}], PRIMARY_LOCAL_MODEL
    )
    return clarified_context


def run_clarification(
    usr_msg: str, decision: Decision, prev_context: str
) -> tuple[str, Decision]:
    cur_msg_ctx = [{"role": "user", "content": usr_msg}]
    clarified_msg = usr_msg
    while decision.route == "ask":
        print("ask route!")

        system = f"""You are internal prompt clarifier of an ai-assistant who finds the ambiguity in user prompt because the the assistant's router was unable to 
        confidently categorize into neither 'chat', 'action', 'reason' nor 'fact' with the given context. So, in the given prompt find why and give a concise two or three sentence (always minimum) feedback to the user on 
        how to improve it for the ai-assistant. Since you and the assistant are one, reply in first person and in friendly and problem solving way. DO NOT BE CONDESCENDING. 
        Candidates that router narrowed down to: {decision.candidates}
        Given Context: {prev_context}"""
        clarify_msg = ask_without_stream(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": "The given user prompt: " + usr_msg},
            ],
            PRIMARY_LOCAL_MODEL,
        )
        print(clarify_msg)

        prev_context += "\nuser: " + usr_msg + "\n" + "assistant: " + clarify_msg
        cur_msg_ctx.append({"role": "assistant", "content": clarify_msg})

        usr_msg = input("Enter message: ")
        if usr_msg.strip() == "/escalate":
            if len(cur_msg_ctx) > 2:
                clarified_msg = run_ctx_clarification(cur_msg_ctx)
            return clarified_msg, Decision(route="escalate")
        cur_msg_ctx.append({"role": "user", "content": usr_msg})
        decision = decide(usr_msg, prev_context, AUTO_ESCALATE)
    clarified_msg = run_ctx_clarification(cur_msg_ctx)
    return clarified_msg, decision


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
    primary_model = PRIMARY_LOCAL_MODEL
    prewarm_cache(primary_model)
    context += estimate_tokens(SYSTEM_PROMPT)
    messages = []
    usr_msg = input("Enter Message: ")
    while usr_msg != "quit":
        if usr_msg == "/model":
            primary_model = prompt_select_model()
        else:
            try:
                prev_context = messages_to_text(messages)
                decision = decide(usr_msg, prev_context, AUTO_ESCALATE)
                clarified_msg = usr_msg
                if decision.route == "ask":
                    clarified_msg, decision = run_clarification(
                        usr_msg, decision, prev_context
                    )
                if decision.route == "local":
                    print("local route")
                    messages.append({"role": "user", "content": clarified_msg})
                    # send conversation with new msg to model
                    res, stats = ask(
                        model=primary_model, messages=messages, system=SYSTEM_PROMPT
                    )
                    print_stats(stats)
                    messages.append({"role": "assistant", "content": res})
                    context = stats["context"]
                elif decision.route == "escalate":
                    print("escalating to Claude...")
                    messages.append({"role": "user", "content": clarified_msg})
                    res, stats = ask(
                        model=ESCALATE_MODEL, messages=messages, system=SYSTEM_PROMPT
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
