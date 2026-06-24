"""
_deepseek.py
------------
Shared helpers for talking to DeepSeek chat models, used by scene_query.py
"""

import os

from openai import OpenAI


def get_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Set the DEEPSEEK_API_KEY environment variable")
    return OpenAI(base_url="https://api.deepseek.com/v1", api_key=api_key)


def strip_think(raw: str) -> str:
    """deepseek-reasoner prefixes its answer with a <think>...</think> block;
    keep only the JSON that follows it."""
    if "</think>" in raw:
        return raw.split("</think>")[-1].strip()
    return raw
