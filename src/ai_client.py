from __future__ import annotations
import os, json, re, requests
from loguru import logger

JSON_RE = re.compile(r"\{.*\}", re.S)   # first JSON object

class AIClient:
    def __init__(self, url: str, api_key_env: str = "GOOGLE_AI_STUDIO_API_KEY", timeout: int = 8, max_retries: int = 2):
        self.url = url
        self.key = os.getenv(api_key_env)
        if not self.key:
            raise EnvironmentError(f"Missing {api_key_env}")
        self.timeout = timeout
        self.max_retries = max_retries

    def _extract_json(self, text: str) -> dict | None:
        try:
            return json.loads(text)
        except Exception:
            m = JSON_RE.search(text)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return None
            return None

    def decide(self, features: dict) -> dict | None:
        prompt = (
            "You are a trading decision engine.\n"
            "Return ONLY a compact JSON object (no prose, no code fences) with keys:\n"
            "{action: buy|sell|flat, entry: float, sl: float, tp: float, confidence: 0-1}.\n"
            "Constraints: Keep RR>=1.5 when possible; use only given numeric features; do not invent prices.\n"
            f"Features: {json.dumps(features)}"
        )
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            # NOTE: Do NOT set response_mime_type for Gemma; it triggers 'JSON mode not enabled'
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 256}
        }

        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(self.url, params={"key": self.key}, json=body, timeout=self.timeout)
                r.raise_for_status()
                text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                parsed = self._extract_json(text)
                if not parsed:
                    raise ValueError(f"Non-JSON reply: {text[:200]}")
                return parsed
            except Exception as e:
                payload = getattr(r, "text", "")[:300]
                logger.warning(f"AI decide error (attempt {attempt}): {e}; payload={payload}")
        return None
