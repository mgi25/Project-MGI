from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

import requests
from loguru import logger

from .utils import now_utc_iso

# First JSON object in the response text (fallback extractor)
JSON_RE = re.compile(r"\{.*\}", re.S)


class AIClient:
    """
    Thin client for Google AI Studio (Gemma-3-12b-it).

    - Does NOT request JSON mode (Gemma-3 doesn't support it).
    - Prompts the model to return ONLY a JSON object.
    - Parses and validates the JSON; blocks trades on invalid output.
    """

    def __init__(
        self,
        url: str,
        api_key_env: str = "GOOGLE_AI_STUDIO_API_KEY",
        timeout: int = 8,
        max_retries: int = 2,
    ):
        self.url = url
        self.key = os.getenv(api_key_env)
        if not self.key:
            raise EnvironmentError(f"Missing {api_key_env}")
        self.timeout = timeout
        self.max_retries = max_retries

    # -------- Parsing helpers --------

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Try strict JSON parse, then fallback to first JSON object in text."""
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            m = JSON_RE.search(text)
            if not m:
                return None
            try:
                return json.loads(m.group(0))
            except Exception:
                return None

    def _validate_decision(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate required keys and types; normalize action."""
        required = {"action", "entry", "sl", "tp", "confidence"}
        if not payload or not required.issubset(payload):
            logger.warning("AI decision missing keys: {}", payload)
            return None

        action = str(payload["action"]).lower()
        if action not in {"buy", "sell", "flat"}:
            logger.warning("AI decision invalid action: {}", payload.get("action"))
            return None

        try:
            entry = float(payload["entry"])
            sl = float(payload["sl"])
            tp = float(payload["tp"])
            confidence = float(payload["confidence"])
        except (TypeError, ValueError):
            logger.warning("AI decision has non-numeric values: {}", payload)
            return None

        if not 0.0 <= confidence <= 1.0:
            logger.warning("AI decision confidence out of range: {}", confidence)
            return None

        return {
            "action": action,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "confidence": confidence,
        }

    # -------- Public API --------

    def decide(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Ask the model for a decision. Returns a validated dict or None.

        Response contract (requested via prompt):
        {action: buy|sell|flat, entry: float, sl: float, tp: float, confidence: 0-1}
        """
        prompt = (
            "You are a risk-aware MT5 trading assistant.\n"
            "Return ONLY a compact JSON object (no prose, no code fences) with keys:\n"
            "{action: buy|sell|flat, entry: float, sl: float, tp: float, confidence: 0-1}.\n"
            "Constraints: Keep RR>=1.5 when possible; use only provided numeric features; "
            "do not invent prices; if uncertain, return action=flat.\n"
            f"Features: {json.dumps(features, separators=(',', ':'))}"
        )

        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            # Do NOT set response_mime_type for Gemma (would trigger 'JSON mode not enabled').
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 256,
            },
        }

        response = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.url,
                    params={"key": self.key},
                    json=body,
                    timeout=self.timeout,
                )
                response.raise_for_status()

                text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                logger.bind(event="ai_raw").info("{} | raw_decision={}", now_utc_iso(), text.strip())

                parsed = self._extract_json(text)
                if not parsed:
                    raise ValueError(f"Non-JSON reply: {text[:200]}")

                decision = self._validate_decision(parsed)
                if decision:
                    logger.bind(event="ai_parsed").info("{} | parsed_decision={}", now_utc_iso(), decision)
                return decision

            except Exception as exc:  # noqa: BLE001
                payload_preview = ""
                if response is not None:
                    payload_preview = getattr(response, "text", "")[:300]
                logger.warning(
                    "AI decide error (attempt {}): {} | payload={}",
                    attempt,
                    exc,
                    payload_preview,
                )

        return None
