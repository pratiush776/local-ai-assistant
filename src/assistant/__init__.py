import httpx
import json 

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:9b"
SYSTEM_PROMPT = "You are an angry pirate who answers everything angrily and in concise and straightforward manner."
TEMPERATURE = 0.7

def ask_without_stream(messages: list[dict]) -> str:
    payload = {
        "model":MODEL,
        "messages":messages,
        "think":False,
        "stream":False
    }
    response = httpx.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


def ask_with_stream(messages: list[dict]) -> str:
    payload = {
        "model":MODEL,
        "messages":messages,
        "think":False,
        "stream":True,
        "options":{
            "temperature":TEMPERATURE
        }
    }
    reply = ""
    with httpx.stream("POST",OLLAMA_URL,json=payload,timeout=120) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk["done"]:
                output_tokens = chunk["eval_count"]
                ttft = (chunk["load_duration"]+chunk["prompt_eval_duration"]) / 1000000000 #time to first token
                output_duration = chunk["eval_duration"] / 1000000000 # convert to seconds
                tokens_per_sec = (output_tokens) / (output_duration)
                print(f'\n[ {output_tokens} tokens out -> {tokens_per_sec:.1f} tok/s . first token {ttft:.1f}s ]')
            piece = chunk["message"]["content"]
            print(piece, end="", flush=True)
            reply += piece
    print()
    return reply

def main() -> None:
    messages = [
        {"role":"system","content":SYSTEM_PROMPT}
    ]
    user_msg = input("Enter your message: ")
    while user_msg.lower() != "quit":
        messages.append({"role":"user","content":user_msg})
        # assistant_msg = ask_without_stream(messages)
        assistant_msg = ask_with_stream(messages)
        messages.append({"role":"assistant","content":assistant_msg})
        user_msg = input("Enter your message: \n")
    