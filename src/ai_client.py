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
            match = JSON_RE.search(text)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except Exception:
                return None

    @staticmethod
    def _to_optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "null", "none"}:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _validate_entry(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        action = str(payload.get("action", "")).lower()
        if action not in {"buy", "sell", "flat"}:
            logger.warning("AI decision invalid action: {}", payload.get("action"))
            return None

        confidence = self._to_optional_float(payload.get("confidence"))
        if confidence is None or not 0.0 <= confidence <= 1.0:
            logger.warning("AI decision confidence invalid: {}", payload.get("confidence"))
            return None

        entry = self._to_optional_float(payload.get("entry"))
        sl = self._to_optional_float(payload.get("sl"))
        tp = self._to_optional_float(payload.get("tp"))
        risk_multiplier = self._to_optional_float(payload.get("risk_multiplier"))
        risk_pct = self._to_optional_float(payload.get("risk_pct") or payload.get("risk_percent"))
        lots = self._to_optional_float(payload.get("lots") or payload.get("lot") or payload.get("volume"))

        if risk_multiplier is not None and risk_multiplier < 0:
            risk_multiplier = 0.0
        if risk_pct is not None and risk_pct < 0:
            risk_pct = 0.0
        if lots is not None and lots < 0:
            lots = 0.0

        if action == "flat":
            return {
                "action": "flat",
                "entry": None,
                "sl": None,
                "tp": None,
                "confidence": confidence,
                "risk_multiplier": risk_multiplier,
                "risk_pct": risk_pct,
                "lots": lots,
            }

        return {
            "action": action,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "confidence": confidence,
            "risk_multiplier": risk_multiplier,
            "risk_pct": risk_pct,
            "lots": lots,
        }

    def _validate_management(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        decision = str(payload.get("decision", "")).lower()
        if decision not in {"hold", "close", "update"}:
            logger.warning("AI management invalid decision: {}", payload.get("decision"))
            return None

        result: Dict[str, Any] = {"decision": decision}
        if decision == "update":
            sl_value = self._to_optional_float(payload.get("sl"))
            if sl_value is not None:
                result["sl"] = sl_value
            tp_value = self._to_optional_float(payload.get("tp"))
            if tp_value is not None:
                result["tp"] = tp_value
        return result

    def _request(self, prompt: str, kind: str) -> Optional[Dict[str, Any]]:
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
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
                logger.bind(event="ai_raw", kind=kind).info("{} | raw={}", now_utc_iso(), text.strip())

                parsed = self._extract_json(text)
                if not parsed:
                    raise ValueError(f"Non-JSON reply: {text[:200]}")

                logger.bind(event="ai_parsed", kind=kind).info("{} | parsed={}", now_utc_iso(), parsed)
                return parsed

            except Exception as exc:  # noqa: BLE001
                payload_preview = ""
                if response is not None:
                    payload_preview = getattr(response, "text", "")[:300]
                logger.warning(
                    "AI {} error (attempt {}): {} | payload={}",
                    kind,
                    attempt,
                    exc,
                    payload_preview,
                )

        return None

    # -------- Public API --------

    def decide_entry(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        prompt = (
            "You are a live MT5 trading assistant for M1 entries.\n"
            "Return ONLY JSON: {action: buy|sell|flat, entry: float|null, sl: float|null, tp: float|null, confidence: 0..1, risk_multiplier?: float>=0, risk_pct?: float>=0, lots?: float>=0}.\n"
            "Bias toward BUY or SELL when there is any directional edge; use FLAT only when signals conflict or are truly absent.\n"
            "Keep entry near current price; place SL/TP where your edge is meaningful. Confidence reflects edge (0..1).\n"
            "Use risk_multiplier to scale the configured base risk, risk_pct to set an absolute equity percentage, or lots to request a specific volume. Leave them absent to use defaults.\n"
            f"Features: {json.dumps(features, separators=(',', ':'))}"
        )

        raw = self._request(prompt, kind="entry")
        if not raw:
            return None
        return self._validate_entry(raw)

    def manage_position(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        prompt = (
            "You manage an open MT5 position given its current state.\n"
            "Return ONLY a JSON object with key 'decision' (hold, update, close).\n"
            "Include numeric 'sl' and/or 'tp' ONLY when decision is update.\n"
            "Use the provided state values directly; do not add commentary.\n"
            f"State: {json.dumps(state, separators=(',', ':'))}"
        )

        raw = self._request(prompt, kind="management")
        if not raw:
            return None
        return self._validate_management(raw)
