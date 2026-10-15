"""LLM factory for Gemini or OpenRouter's OpenAI-compatible API."""

import os

from langchain_openai import ChatOpenAI

from common.schemas import AUTO_APPROVE_THRESHOLD, ESCALATE_THRESHOLD


def review_system_prompt() -> str:
    """Shared confidence calibration for all four review graphs."""
    return (
        "You are a senior code reviewer. Return a complete structured PRAnalysis. "
        "Inspect the changed code for correctness, security, tests, and migration "
        "or rollout risks. Confidence is your certainty that your review is complete "
        "and correct; it is not the probability that the patch is bug-free. "
        "Calibrate conservatively against unresolved context. Reserve confidence "
        f">= {AUTO_APPROVE_THRESHOLD:.2f} for mechanical changes with no behavior, "
        "schema, authentication, or security impact and no open questions. "
        f"For a behavior or schema change with a manageable open question, use "
        f"{ESCALATE_THRESHOLD:.2f} <= confidence < {AUTO_APPROVE_THRESHOLD:.2f}. "
        f"If authentication, secrets, unsafe query construction, or several serious "
        f"findings leave the threat model or implementation intent unclear, use "
        f"confidence < {ESCALATE_THRESHOLD:.2f}. Explain the missing information "
        "in confidence_reasoning. For low confidence, provide 2–4 specific "
        "escalation_questions naming the relevant file and function or section."
    )


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if gemini_key:
        api_key = gemini_key
        default_model = "gemini-3.5-flash-lite"
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
