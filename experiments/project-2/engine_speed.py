import json

import httpx

MODELS = ["qwen3.5:9b","qwen3.5:9b-q4_K_M"]
PROMPT = "Explain in about 150 words why the sky is blue."
RUNS = 3

def call(model:str,prompt:str) -> dict:
    payload = {
        "model":model,
        "messages":[
            {
                "role":"user",
                "content": prompt
            }
        ],
        "think":False,
        "stream":True
    }

    stats = {}
    try:
        with httpx.stream("POST",url="http://localhost:11434/api/chat",json=payload,timeout=300) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk["done"]:
                    stats["out_toks"] = chunk["eval_count"]
                    stats["in_toks"] = chunk["prompt_eval_count"]
                    stats["in_duration"] = chunk["prompt_eval_duration"]/1000000000
                    stats["load_duration"]= chunk["load_duration"]/1000000000
                    stats["out_duration"] = chunk["eval_duration"] / 1000000000

                    stats["ttft"] = round(stats["load_duration"]+stats["in_duration"],2)
                    stats["out_tps"] = round(stats["out_toks"]/stats["out_duration"],2)
                    stats["in_tps"] = round(stats["in_toks"]/stats["in_duration"],2)
    except httpx.HTTPError as e:
        print("Network Error: ", e)
    return stats
    

for model in MODELS:
    print(f"FOR {model}")
    print("\tWarm_up")
    call(model,"hi")
    for i in range(RUNS):
        stats = call(model,PROMPT)

        print(f"\tRun {i} {stats["out_tps"]} tok/s out . {stats["in_tps"]} tok/s in . {stats["ttft"]}s to first tok")