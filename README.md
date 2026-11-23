# Local-first personal assistant

I’m learning applied AI engineering by building a personal assistant. The current prototype runs chat, file search, and an approval-gated calendar and reminder agent on Ollama and SQLite. The CLI and Gradio UI share the same turn logic and write local traces. Claude is available as an explicit escalation.

## Run locally

Start Ollama and pull the models you’re missing. Run the setup from the repo root with `uv` installed.

```sh
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
uv sync --locked
uv run assistant-gui
```

Upload a PDF, Markdown, or text file to index it, then ask a specific question about it. The CLI is available with `uv run assistant`. To send a turn to Claude, use `/claude <message>` in the GUI or `/escalate` before your next CLI message. Local use works without an Anthropic API key.

The checked-out code provides the app. The persona fixtures, prepared index, and saved eval runs live under the git-ignored `data/` directory in the learning workspace. If you have those fixtures, index the demo corpus with:

```sh
uv run python -m assistant.knowledge.ingest data/assets/persona/files data/assets/persona/documents
```

The agent seeds its demo calendar when the fixture is available. `uv run evals --agent` uses the local task and calendar fixtures and runs each task in its own in-memory database.

## Earlier project notes

project 0: simple api call to local model running in Ollama

project 1: streaming model gateway for local and cloud chat. The current escalation commands are in the setup instructions above.

project 2: quick decisions with confidence - basically, how do you know the category (action, fact, chat, reason) and confidence of the classification of a request accurately via the local model?

- the fist question was _how do you even get a confidence?_ three ways: (a) ask the model how sure it is (verbalized), (b) read the token logprobs myself, or (c) let a dedicated decision model (tev/nimble) hand me the decision. asking, (a), was useless — it said ~0.97 confident to everything, no spread to threshold on. the honest signal is the raw logprob of the single label token the model emits: run the classify prompt, stop after one token, read the probability mass on `action`/`fact`/`reason`/`chat`. that number is the confidence everything else is built on. no JSON, no `format` — that fights qwen's template and buries the number I want. (had to rename `reasoning`→`reason` so every label is exactly one token to read cleanly.)

- synthetic data lied to me. my own exam said 85%; the day I tested on real, balanced data (gsm8k / nq_open / dolly / taskmaster, 200 each) it was 63.5%. never grade yourself on questions you wrote.

- the model wasn't dumb, it was overconfident. it kept categorizing `fact` and `reason` into `chat` — because I'd asked it to judge what it _doesn't know_, the one thing a small model can't do. the fix wasn't a bigger model, it was telling it to stop trusting itself ("call it fact even if you think you know the answer"). few-shot + that one cue took it 63.5 → 88.6, reason 38 → 100.

- one model, not two. I benchmarked a dedicated decision model (tev 0.8B & 4B, and nimble 9B); on a neutral 1000-row test they _tied_ qwen — and "tied" means McNemar said so (paired test, same questions, not me eyeballing two numbers). they also ate the memory on my laptop which I needed for context eventually. so qwen did both; thus the winner.

- the stats did real work. I sized the tests with Wilson intervals (200/label ≈ ±5.5%; at n=200 a 4-point gap is still noise — which is exactly why the early small runs looked inconclusive). and the lesson that stuck: by confidence intervals tev-0.8B looked level with the 4B (they overlapped), but McNemar — which judges the _same_ questions both models answered, not two separate error bars — showed 0.8B was genuinely worse. paired testing sees what comparing error bars can't.

- the threshold barely matters. at 0.65 I keep 95% local at 91% right — but the mistakes are as confident as the hits, so no knob saves me. the classifier's accuracy is the ceiling. worth knowing where the real constraint is.

- I spend the money, not the model. escalating to Claude is always my call; when the router is torn it asks me, because only I can resolve "what did you mean" and only a bigger model resolves "this is hard" — and the router can't tell those two apart.
