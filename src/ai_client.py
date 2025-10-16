from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import requests
from loguru import logger


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

    def _build_payload(self, features: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            "You are a trading decision engine. Return strict JSON: "
            "{action: buy|sell|flat, entry: float, sl: float, tp: float, confidence: 0-1}. "
            "Use only provided features; never invent prices. Keep RR>=1.5 when possible.\n"
            f"Features: {json.dumps(features, sort_keys=True)}"
        )
        return {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }

    def decide(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not features:
            logger.debug("Skipping AI call: empty feature set")
            return None

        body = self._build_payload(features)
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
                decision = json.loads(text)
                if not isinstance(decision, dict):
                    raise ValueError("Model response is not a JSON object")
                return decision
            except Exception as exc:  # noqa: BLE001 - keep broad for retries
                logger.warning(
                    "AI decide error (attempt {}/{}): {}",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
        return None
