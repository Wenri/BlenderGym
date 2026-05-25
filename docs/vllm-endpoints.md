# Hosted vllm Endpoints

Two OpenAI-compatible vllm servers are exposed publicly via Cloudflare. Both
serve Qwen3-family reasoning models with a 262K-token context window.

| Host                        | Model                                   | Context  | Backing GPUs    |
| --------------------------- | --------------------------------------- | -------- | --------------- |
| `https://vlm1.wenri.me/v1`  | `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4`      | 262144   | 4×H100 (vega)   |
| `https://vlm2.wenri.me/v1`  | `Qwen/Qwen3.6-27B-FP8`                  | 262144   | 2×L40S (saturn) |

The serving commands live in `pyproject.toml` under
`[tool.pixi.feature.vllm.tasks]` (`serve-qwen35`, `serve-qwen36-27b`).

## Authentication

All requests require a Bearer token. Set it once in your shell and reference it
as `$VLLM_API_KEY`:

```bash
export VLLM_API_KEY=...   # ask the maintainer; never commit this value
```

The token is shared across both endpoints.

## Quick test (curl)

```bash
# List the model loaded on each host
curl -sS -H "Authorization: Bearer $VLLM_API_KEY" https://vlm1.wenri.me/v1/models | jq
curl -sS -H "Authorization: Bearer $VLLM_API_KEY" https://vlm2.wenri.me/v1/models | jq

# One-shot chat completion
curl -sS https://vlm2.wenri.me/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.6-27B-FP8",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 4096,
    "temperature": 0.2
  }' | jq -r '.choices[0].message.content'
```

## Python (OpenAI SDK)

The servers are OpenAI-compatible, so the `openai` SDK works as-is — just point
`base_url` at the host's `/v1` path.

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["VLLM_API_KEY"],
    base_url="https://vlm2.wenri.me/v1",
)

resp = client.chat.completions.create(
    model="Qwen/Qwen3.6-27B-FP8",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=4096,
    temperature=0.2,
)
print(resp.choices[0].message.content)
```

A ready-to-run example that hits both endpoints lives at
[`query_vllm.py`](../query_vllm.py) in the repo root:

```bash
VLLM_API_KEY=... pixi run -- python query_vllm.py "your prompt here"
```

## Notes

- **Both models are reasoning models.** They emit a `<think>...</think>` trace
  before the visible answer. vllm is launched with `--reasoning-parser qwen3`,
  so the trace lands in `choices[0].message.reasoning_content` (or is dropped,
  depending on the response) and the user-facing reply lands in `.content`.
  Pick `max_tokens` large enough to cover both — **4096 is a safe default**;
  smaller caps (e.g. 150) routinely leave `.content` empty because the trace
  exhausts the budget. The bigger 397B model on `vlm1` tends to use a longer
  trace than the 27B on `vlm2`.

- **OpenAI-compatible surface.** `chat/completions`, `completions`,
  `embeddings` (when supported by the model), and `models` all work. Streaming
  via `stream=true` works too.

- **Long context (262K).** Useful for big-prompt experiments, but every long
  request pays linearly more compute on the server side — don't pad
  unnecessarily.

- **Reachability.** The public hosts route through Cloudflare and add ~50 ms of
  WAN/TLS overhead vs. talking to the boxes directly. If you're on the
  Polytechnique VPN and care about latency, you can also SSH-tunnel to
  `saturn-lix.polytechnique.fr:8000` / `vega-lix.polytechnique.fr:8000`
  directly (`-L /tmp/vllm-*.sock:localhost:8000 <host>`).
