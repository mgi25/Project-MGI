from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import requests
from loguru import logger

from .utils import now_utc_iso


class AIClient:
    """Thin wrapper over Google AI Studio's Gemma models."""

    def __init__(
        self,
        url: str,
        api_key_env: str = "GOOGLE_AI_STUDIO_API_KEY",
        timeout: int = 8,
        max_retries: int = 2,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries
        self.key = os.getenv(api_key_env)
        if not self.key:
            raise EnvironmentError(f"Missing {api_key_env}")

    def decide(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        prompt = (
            "You are a risk-aware MT5 trading assistant. Return ONLY a JSON object "
            "with the keys action, entry, sl, tp, confidence. No prose, no code "
            "fences. Use the market features verbatim; never speculate prices.\n"
            f"Features: {json.dumps(features, separators=(',', ':'))}"
        )
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 256,
            },
        }
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
                decision = self._parse_decision(text)
                logger.bind(event="ai_raw").info("{} | raw_decision={}", now_utc_iso(), text.strip())
                if decision:
                    logger.bind(event="ai_parsed").info("{} | parsed_decision={}", now_utc_iso(), decision)
                return decision
            except Exception as exc:  # noqa: BLE001
                payload_preview = ""
                if "response" in locals():
                    payload_preview = getattr(response, "text", "")[:300]
                logger.warning(
                    "AI decide error (attempt {}): {} | payload={}",
                    attempt,
                    exc,
                    payload_preview,
                )
        return None

    def _parse_decision(self, text: str) -> Optional[Dict[str, Any]]:
        import re

        if not text:
            return None

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            logger.warning("AI response missing JSON object: {}", text[:120])
            return None

        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            logger.warning("AI JSON decode error: {}", exc)
            return None

        required = {"action", "entry", "sl", "tp", "confidence"}
        if not required.issubset(payload):
            logger.warning("AI decision missing keys: {}", payload)
            return None

        action = str(payload["action"]).lower()
        if action not in {"buy", "sell", "flat"}:
            logger.warning("AI decision invalid action: {}", payload["action"])
            return None

        try:
            entry = float(payload["entry"])
            sl = float(payload["sl"])
            tp = float(payload["tp"])
            confidence = float(payload["confidence"])
        except (TypeError, ValueError):
            logger.warning("AI decision has non-numeric prices: {}", payload)
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
