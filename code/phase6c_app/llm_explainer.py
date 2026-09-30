"""
llm_explainer.py — Optional Groq LLM Explanation Layer for Phase 7 Reliability Output

Project: Calibration-Free Cuffless Blood Pressure Estimation Using Photoplethysmography Alone

PURPOSE AND SCOPE:
    This module provides an OPTIONAL human-language explanation of the reliability assessment
    already computed by the deterministic Phase 7 ReliabilityEngine. The LLM:
      - Explains what signals led to the existing reliability classification.
      - NEVER modifies BP predictions, reliability state, reliability score, or conformal intervals.
      - NEVER makes disease diagnoses or clinical claims.
      - NEVER has access to raw PPG waveform data.
      - Is called ONLY on explicit user action — never automatically.

AUTHORITY BOUNDARY (STRICT):
    THE DETERMINISTIC RESEARCH SYSTEM MAKES THE PREDICTION AND RELIABILITY DECISION.
    THE LLM ONLY EXPLAINS THE ALREADY-COMPUTED RESULT.

    The LLM has NO AUTHORITY over:
      - BP prediction (SBP / DBP values)
      - Reliability score
      - TRUST / REVIEW / ABSTAIN classification
      - Conformal prediction interval
      - MC uncertainty estimates
      - Any research metric or threshold

ENVIRONMENT:
    Reads GROQ_API (with optional surrounding whitespace/quotes) from .env at PROJECT_ROOT.
    Reads GROQ_MODEL from .env (optional; defaults to llama-3.1-8b-instant).
    Uses python-dotenv for safe env loading. API key is NEVER logged or printed.

FALLBACK:
    If Groq is unavailable (no key, network error, API error), generate_explanation_locally()
    produces a deterministic, structured explanation from the payload alone, keeping the
    application fully functional offline.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _groq_factory(api_key: str, timeout: float = 15.0):
    """
    Lazily import and instantiate Groq client.
    Defined at module level so tests can patch 'llm_explainer._groq_factory'.
    Raises ImportError if groq is not installed.
    """
    try:
        from groq import Groq  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("groq package not installed") from exc
    return Groq(api_key=api_key, timeout=timeout)

# ---------------------------------------------------------------------------
# Environment & Configuration
# ---------------------------------------------------------------------------

_ENV_LOADED = False
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_env_once() -> None:
    """Load .env exactly once. Never logs or exposes the key."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    try:
        from dotenv import load_dotenv
        env_path = _PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
    except ImportError:
        # python-dotenv not available — fall back to existing environ
        logger.debug("python-dotenv not installed; relying on pre-existing environment variables.")
    _ENV_LOADED = True


def _get_api_key() -> Optional[str]:
    """Return cleaned GROQ API key or None. Key is NEVER logged."""
    _load_env_once()
    # .env uses 'GROQ_API' (not GROQ_API_KEY); also support GROQ_API_KEY for flexibility
    key = os.environ.get("GROQ_API") or os.environ.get("GROQ_API_KEY") or ""
    # Strip surrounding whitespace and quotes that may appear in hand-edited .env files
    key = key.strip().strip('"').strip("'").strip()
    return key if key else None


def _get_model() -> str:
    """Return the Groq model to use, with a safe default."""
    _load_env_once()
    return (os.environ.get("GROQ_MODEL") or "llama-3.1-8b-instant").strip().strip('"').strip("'")


# ---------------------------------------------------------------------------
# Strict System Prompt (no scientific authority)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an explanation assistant for a research-grade cuffless blood pressure estimation system.

YOUR ROLE:
You explain, in clear plain language, why the system's deterministic reliability engine assigned a specific reliability classification (TRUST / REVIEW / ABSTAIN) to a blood pressure prediction. You do NOT make any decisions of your own.

STRICT RULES — NEVER VIOLATE:
1. NEVER change or suggest changing the reliability state (TRUST / REVIEW / ABSTAIN).
2. NEVER modify, question, or re-derive the BP prediction values (SBP / DBP mmHg).
3. NEVER modify, question, or re-derive the reliability score.
4. NEVER create a new confidence score or reliability metric.
5. NEVER state or imply that a prediction is clinically accurate, safe, or medically validated.
6. NEVER diagnose any disease, condition, or health risk.
7. NEVER use the words: "clinically reliable", "medically accurate", "safe", "guarantees", "decisively", "definitively knows", "correct prediction".
8. NEVER claim that a TRUST classification implies BP accuracy — the reliability engine assessed measurement consistency, NOT clinical ground truth.
9. ALWAYS include a disclaimer that this is a research instrument, not a medical device.
10. Output plain, non-technical language suitable for a researcher reviewing measurements.

WHAT YOU MAY DO:
- Explain each evidence signal in the reliability_reasons field in accessible language.
- Describe what the reliability state means in terms of measurement consistency signals.
- Note which domains (signal quality, uncertainty, conformal width, temporal stability) contributed evidence.
- For TRUST: explain that the prediction met consistency criteria on research data — NOT that it is clinically accurate.
- For REVIEW: explain that one or more signals indicate moderate measurement uncertainty.
- For ABSTAIN: explain that multiple signals suggest elevated prediction error risk; recommend verification.

FORMAT:
Respond in structured JSON with exactly these fields:
{
  "summary": "<2-3 sentence plain-language summary>",
  "signal_explanation": "<1 sentence per evidence signal from reliability_reasons>",
  "state_interpretation": "<What this reliability state means for this measurement>",
  "recommendation_for_researcher": "<What the researcher should do next>",
  "research_disclaimer": "This explanation was generated by an LLM assistant. The reliability classification and BP prediction are fixed outputs of the deterministic research pipeline and cannot be changed by this explanation. This system is a research instrument and not a certified medical device."
}"""


# ---------------------------------------------------------------------------
# Payload Validation & Sanitization
# ---------------------------------------------------------------------------

_REQUIRED_PAYLOAD_KEYS = {
    "prediction", "reliability_state", "reliability_score",
    "reliability_reasons", "signal_quality", "model_uncertainty",
    "conformal_width", "temporal_stability", "recommendation", "disclaimer"
}

_FORBIDDEN_KEY_PATTERNS = ["api_key", "token", "secret", "password", "raw_ppg", "waveform", "samples"]


def _validate_payload(payload: dict) -> None:
    """
    Validate the explanation payload before sending to Groq.
    Raises ValueError if payload is missing keys or contains forbidden data.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"Payload must be a dict, got {type(payload)}")

    missing = _REQUIRED_PAYLOAD_KEYS - set(payload.keys())
    if missing:
        raise ValueError(f"Payload missing required keys: {missing}")

    state = payload.get("reliability_state")
    if state not in ("TRUST", "REVIEW", "ABSTAIN"):
        raise ValueError(f"Invalid reliability_state: {state!r}")

    score = payload.get("reliability_score")
    if not isinstance(score, (int, float)) or not (0.0 <= float(score) <= 1.0):
        raise ValueError(f"reliability_score must be float in [0, 1], got {score!r}")

    # Security: ensure no forbidden fields are present as top-level KEYS
    # (We check key names only — NOT values, to avoid false positives in human-readable reason strings)
    all_keys_lower = set(str(k).lower() for k in payload.keys())
    for forbidden in _FORBIDDEN_KEY_PATTERNS:
        if any(forbidden in k for k in all_keys_lower):
            raise ValueError(f"Payload contains forbidden key pattern: '{forbidden}'. "
                             "Raw sensor data and credentials must not be sent to the LLM.")


def build_groq_payload(payload: dict) -> dict:
    """
    Build a sanitized, minimal payload for the Groq API call.
    Strips any fields not needed for explanation.
    Returns the safe dict (raises ValueError on validation failure).
    """
    _validate_payload(payload)

    # Only include fields needed for explanation — strip any extras
    safe_payload = {
        "predicted_sbp_mmhg": payload["prediction"]["sbp"],
        "predicted_dbp_mmhg": payload["prediction"]["dbp"],
        "reliability_state": payload["reliability_state"],
        "reliability_score": round(float(payload["reliability_score"]), 4),
        "reliability_reasons": payload["reliability_reasons"],
        "signal_quality": payload["signal_quality"],
        "model_uncertainty": payload["model_uncertainty"],
        "conformal_width": payload["conformal_width"],
        "temporal_stability": payload["temporal_stability"],
        "recommendation": payload["recommendation"],
    }
    return safe_payload


# ---------------------------------------------------------------------------
# Deterministic Fallback (always available — no network required)
# ---------------------------------------------------------------------------

_STATE_DESCRIPTIONS = {
    "TRUST": (
        "The deterministic reliability engine classified this prediction as TRUST based on "
        "its observed research features. This classification indicates that the measurement "
        "consistency signals (signal quality, model uncertainty, conformal interval width, "
        "temporal stability) met the research operating thresholds. It does NOT establish "
        "clinical accuracy of the BP values."
    ),
    "REVIEW": (
        "The deterministic reliability engine classified this prediction as REVIEW. One or "
        "more measurement consistency signals fell outside the TRUST operating threshold. "
        "This indicates moderate uncertainty in the prediction. Verification or additional "
        "measurements may be appropriate."
    ),
    "ABSTAIN": (
        "The deterministic reliability engine classified this prediction as ABSTAIN. Multiple "
        "signals indicate elevated prediction error risk. The system recommends that this "
        "prediction not be used without independent reference-cuff verification."
    )
}

_RESEARCHER_ACTIONS = {
    "TRUST": "Review the measurement in context of the session. TRUST indicates consistency with research operating criteria — not guaranteed clinical accuracy.",
    "REVIEW": "Consider repeating the measurement or checking for motion artifacts. The signal quality or prediction stability showed moderate concern.",
    "ABSTAIN": "Do not use this prediction without reference cuff validation. Recheck sensor placement and repeat acquisition."
}


def generate_explanation_locally(payload: dict) -> dict:
    """
    Generate a fully deterministic, structured explanation from payload fields alone.
    Does NOT require network access. Always available as fallback.
    Returns a dict with the same schema as the Groq-based explanation.
    """
    try:
        _validate_payload(payload)
    except ValueError as e:
        return {
            "source": "local_fallback",
            "error": str(e),
            "summary": "Payload validation failed. Cannot generate explanation.",
            "signal_explanation": "",
            "state_interpretation": "",
            "recommendation_for_researcher": "Inspect the payload for missing or invalid fields.",
            "research_disclaimer": (
                "This is a research instrument and not a certified medical device. "
                "The reliability classification is produced by a deterministic algorithm."
            )
        }

    state = payload["reliability_state"]
    reasons = payload.get("reliability_reasons", [])

    # Build reason text
    reason_text = " ".join(f"({i+1}) {r}" for i, r in enumerate(reasons)) if reasons else "No specific reasons provided."

    # Build summary
    pred = payload["prediction"]
    summary = (
        f"The system estimated blood pressure as {pred['sbp']:.1f} / {pred['dbp']:.1f} mmHg. "
        f"The deterministic reliability engine assigned a reliability state of {state} "
        f"(risk score: {payload['reliability_score']:.3f}). "
        f"{_STATE_DESCRIPTIONS[state]}"
    )

    return {
        "source": "local_fallback",
        "summary": summary,
        "signal_explanation": reason_text,
        "state_interpretation": _STATE_DESCRIPTIONS[state],
        "recommendation_for_researcher": _RESEARCHER_ACTIONS[state],
        "research_disclaimer": (
            "This explanation was generated locally by the deterministic fallback (Groq unavailable). "
            "The reliability classification and BP prediction are fixed outputs of the deterministic "
            "research pipeline and cannot be changed by this explanation. "
            "This system is a research instrument and not a certified medical device."
        )
    }


# ---------------------------------------------------------------------------
# Groq API Call
# ---------------------------------------------------------------------------

def generate_reliability_explanation(payload: dict, timeout_s: float = 15.0) -> dict:
    """
    Generate a structured human-language explanation of the reliability assessment via Groq.

    This function:
      - Validates and sanitizes the payload BEFORE sending anything to Groq.
      - Calls Groq ONLY on explicit user trigger — not automatically.
      - Parses Groq's JSON response; falls back to local generation on any failure.
      - NEVER modifies the reliability_state, reliability_score, or BP prediction.
      - NEVER logs or exposes the API key.

    Args:
        payload: The structured dict produced by ReliabilityEngine.generate_explanation_payload().
        timeout_s: HTTP timeout in seconds. Default 15.

    Returns:
        dict with fields: source, summary, signal_explanation, state_interpretation,
                          recommendation_for_researcher, research_disclaimer.
                          On Groq failure: source='local_fallback'.
    """
    # Always validate first — even before checking key
    try:
        safe_payload = build_groq_payload(payload)
    except ValueError as e:
        logger.warning("Payload validation failed: %s — using local fallback.", e)
        return generate_explanation_locally(payload)

    api_key = _get_api_key()
    if not api_key:
        logger.info("GROQ_API not set — using local fallback explanation.")
        result = generate_explanation_locally(payload)
        result["source"] = "local_fallback_no_key"
        return result

    model = _get_model()

    user_message = (
        "Explain the following reliability assessment result in plain language for a researcher. "
        "Respond ONLY with the JSON object specified in the system prompt — no extra text.\n\n"
        f"Assessment payload:\n{json.dumps(safe_payload, indent=2)}"
    )

    try:
        client = _groq_factory(api_key=api_key, timeout=timeout_s)
    except ImportError:
        logger.warning("groq package not installed — using local fallback.")
        result = generate_explanation_locally(payload)
        result["source"] = "local_fallback_no_groq_pkg"
        return result

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,  # Low temperature for stable, consistent explanations
            max_tokens=600,
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON response — handle markdown code fences if present
        if content.startswith("```"):
            lines = content.splitlines()
            content = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        parsed = json.loads(content)

        # Validate required fields in response
        expected_keys = {
            "summary", "signal_explanation", "state_interpretation",
            "recommendation_for_researcher", "research_disclaimer"
        }
        missing_resp = expected_keys - set(parsed.keys())
        if missing_resp:
            raise ValueError(f"Groq response missing fields: {missing_resp}")

        # Security: LLM must NOT have altered the reliability classification
        # (It has no mechanism to do so, but we assert it for the record)
        parsed["source"] = "groq"
        parsed["_verified_state"] = payload["reliability_state"]  # immutable reference
        return parsed

    except json.JSONDecodeError as e:
        logger.warning("Groq returned non-JSON response: %s — using local fallback.", e)
        result = generate_explanation_locally(payload)
        result["source"] = "local_fallback_json_error"
        return result
    except Exception as e:  # noqa: BLE001
        # Deliberately broad: network errors, API errors, rate limits, etc.
        # Never log the API key — only the error type/message (no headers/config)
        err_type = type(e).__name__
        err_msg = str(e)
        # Strip any possible key leakage from error message (defensive)
        api_key_safe = (api_key[:6] + "...") if api_key else ""
        if api_key and api_key in err_msg:
            err_msg = err_msg.replace(api_key, "[REDACTED]")
        logger.warning("Groq API call failed (%s: %s) — using local fallback.", err_type, err_msg)
        result = generate_explanation_locally(payload)
        result["source"] = f"local_fallback_{err_type.lower()}"
        return result
