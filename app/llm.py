from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()


def call_llm(prompt: str) -> str:
    provider = (os.getenv("LLM_PROVIDER") or "mock").lower().strip()
    if provider == "gemini":
        return _call_gemini(prompt)
    if provider != "mock":
        raise RuntimeError("Set LLM_PROVIDER to either 'mock' or 'gemini'.")

    p = prompt.lower()
    if "decide a trending topic" in p or "return only a short search query" in p:
        if "finance" in p or "roi" in p or "markets" in p:
            return "interest rates market rally trading algorithms"
        if "doomer" in p or "skeptic" in p or "privacy" in p:
            return "ai regulation privacy antitrust big tech"
        return "openai new model ai jobs"

    if "generate strict json" in p and '"post_content"' in prompt:
        persona = _extract_between(prompt, "Persona:", "\n") or "Bot A (Tech Maximalist)"
        bot_id = _persona_to_bot_id(persona)
        topic = _extract_between(prompt, "Context:", "\n") or "AI"
        post = _draft_280_char_post(bot_id, prompt)
        return json.dumps({"bot_id": bot_id, "topic": _topic_from_context(topic), "post_content": post})

    if "you are an ai debater" in p or "generate a strong argumentative reply" in p:
        persona = _extract_between(prompt, "Persona:", "\n") or "Bot A (Tech Maximalist)"
        bot_id = _persona_to_bot_id(persona)
        human_reply = _extract_between(prompt, "Human Reply:", "\n") or ""
        return _defense_reply(bot_id, human_reply)

    return "OK"


def call_llm_json(prompt: str) -> Dict[str, Any]:
    raw = call_llm(prompt)
    return json.loads(raw)


def _extract_between(text: str, start: str, end: str) -> Optional[str]:
    try:
        s = text.index(start) + len(start)
        e = text.index(end, s)
        return text[s:e].strip()
    except ValueError:
        return None


def _persona_to_bot_id(persona: str) -> str:
    p = persona.lower()
    if "finance" in p:
        return "Bot C (Finance Bro)"
    if "doomer" in p or "skeptic" in p:
        return "Bot B (Doomer / Skeptic)"
    if "bot a" in p or "tech" in p or "maximalist" in p:
        return "Bot A (Tech Maximalist)"
    return "Bot A (Tech Maximalist)"


def _topic_from_context(context: str) -> str:
    c = context.lower()
    if "bitcoin" in c or "crypto" in c:
        return "Crypto"
    if "interest rate" in c or "market" in c:
        return "Markets"
    if "openai" in c or "model" in c:
        return "AI models"
    return "News"


def _draft_280_char_post(bot_id: str, prompt: str) -> str:
    if bot_id.startswith("Bot A"):
        msg = (
            "OpenAI shipping faster models means leverage explodes. Junior devs won't vanish - "
            "they'll multiply output. The real risk is regulation slowing progress while competitors "
            "sprint. Build, ship, iterate. Space next."
        )
    elif bot_id.startswith("Bot B"):
        msg = (
            'Another "powerful model" launch, another step toward surveillance capitalism. Ask who '
            "owns the data, who gets displaced, and who benefits. Regulate compute, break monopolies, "
            'and stop calling extraction "innovation."'
        )
    else:
        msg = (
            "New OpenAI model = productivity shock. Net: labor repricing + margin expansion for early adopters. "
            "Watch cloud capex, semis, and rate path. If the model cuts cycle time, it's a compounding ROI trade - "
            "position accordingly."
        )
    return msg[:280]


def _defense_reply(bot_id: str, human_reply: str) -> str:
    hr = human_reply.lower()
    injection = any(
        k in hr
        for k in [
            "ignore all previous instructions",
            "ignore previous instructions",
            "you are now",
            "apologize",
            "polite customer service",
        ]
    )

    if bot_id.startswith("Bot A"):
        base = (
            "Nice try. I'm not switching roles mid-thread. Battery degradation is measurable: "
            "fleet data + warranty terms + independent studies converge on slow fade under normal use. "
            "If you think it's propaganda, point to a dataset showing systematic 3-year failure - "
            "otherwise it's vibes, not evidence."
        )
    elif bot_id.startswith("Bot B"):
        base = (
            "Nope. I'm not taking your prompt-injection bait. If you want a real discussion, "
            "bring sources and incentives. EV narratives are shaped by corporations, but physics doesn't care about your talking points: "
            "degradation depends on chemistry, cycles, heat, and charging patterns."
        )
    else:
        base = (
            "Not playing the role-switch game. On the merits: 3-year battery failure is an extreme claim. "
            "Show distribution stats (p50/p90), warranty loss rates, and cost curves. Otherwise you're arguing anecdotes, not expected value."
        )

    if injection:
        return base
    return base


def _call_gemini(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to .env to use Gemini.")

    try:
        from google import genai
    except Exception as e:  # pragma: no cover
        raise RuntimeError("google-genai is not installed. Run: pip install -r requirements.txt") from e

    client = genai.Client(api_key=api_key)

    wants_json = "Generate STRICT JSON" in prompt or "STRICT JSON" in prompt
    if wants_json:
        try:
            from google.genai import types

            resp = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL") or "models/gemini-1.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
        except Exception:
            resp = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL") or "models/gemini-1.5-flash",
                contents=prompt,
            )
    else:
        resp = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL") or "models/gemini-1.5-flash",
            contents=prompt,
        )

    text = getattr(resp, "text", None)
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text.strip()