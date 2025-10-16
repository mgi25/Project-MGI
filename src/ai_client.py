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

    def decide(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        prompt = (
            "You are a trading decision engine. Return strict JSON: "
            "{action: buy|sell|flat, entry: float, sl: float, tp: float, confidence: 0-1}. "
            "Use only provided features; never invent prices.\n"
            f"Features: {json.dumps(features)}"
        )
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                # IMPORTANT: snake_case key for AI Studio
                "response_mime_type": "application/json"
            },
        }
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(
                    self.url,
                    params={"key": self.key},
                    json=body,
                    timeout=self.timeout,
                )
                r.raise_for_status()
                text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text)
            except Exception as e:
                logger.warning(
                    f"AI decide error (attempt {attempt}): {e}; payload={getattr(r, 'text', '')[:300]}"
                )
        return None
