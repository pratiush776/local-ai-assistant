import httpx

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:9b"


def ask(messages: list[dict]) -> str:
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


def main() -> None:
    messages = []
    user_msg = input("Enter your message: ")
    while user_msg.lower() != "quit":
        messages.append({"role":"user","content":user_msg})
        assistant_msg = ask(messages)
        print(assistant_msg)
        messages.append({"assistant":"assistant","content":assistant_msg})
        user_msg = input("Enter your message: \n")
    