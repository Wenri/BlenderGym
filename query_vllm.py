"""Query both public vllm endpoints (Qwen3.5-397B on vlm1, Qwen3.6-27B on vlm2).

Usage:

    VLLM_API_KEY=... python query_vllm.py "your prompt here"
"""

import os
import sys

from openai import OpenAI


ENDPOINTS = [
    {
        "name": "vlm1",
        "base_url": "https://vlm1.wenri.me/v1",
        "model": "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4",
    },
    {
        "name": "vlm2",
        "base_url": "https://vlm2.wenri.me/v1",
        "model": "Qwen/Qwen3.6-27B-FP8",
    },
]


def main() -> int:
    api_key = os.environ.get("VLLM_API_KEY")
    if not api_key:
        print("Set VLLM_API_KEY before running.", file=sys.stderr)
        return 1

    prompt = " ".join(sys.argv[1:]) or "In one sentence, what is the Blender bpy API used for?"

    for ep in ENDPOINTS:
        client = OpenAI(api_key=api_key, base_url=ep["base_url"], timeout=180.0)
        print(f"=== {ep['name']} ({ep['model']}) ===")
        try:
            resp = client.chat.completions.create(
                model=ep["model"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,  # Qwen3 thinking models need headroom for <think>...</think>
                temperature=0.2,
            )
        except Exception as exc:
            print(f"[{ep['name']}] request failed: {exc}\n")
            continue

        msg = resp.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            print(f"[thinking]\n{reasoning.strip()}\n")
        print((msg.content or "").strip())

        usage = resp.usage
        print(f"[usage: prompt={usage.prompt_tokens} completion={usage.completion_tokens}]\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
