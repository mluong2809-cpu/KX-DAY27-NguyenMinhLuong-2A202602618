"""LLM factory for Gemini or OpenRouter's OpenAI-compatible API."""

import os

from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if gemini_key:
        api_key = gemini_key
        default_model = "gemini-3.8-flash"
        default_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    elif openrouter_key:
        api_key = openrouter_key
        default_model = "openai/gpt-4o-mini"
        default_base_url = "https://openrouter.ai/api/v1"
    else:
        raise RuntimeError("Set GEMINI_API_KEY or OPENROUTER_API_KEY in .env")
    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL") or default_model,
        base_url=os.environ.get("LLM_BASE_URL") or default_base_url,
        api_key=api_key,
        temperature=temperature,
    )
