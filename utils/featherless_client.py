"""
utils/featherless_client.py
Thin wrapper around the Featherless AI API.
"""

import logging
import requests
from typing import Dict

logger = logging.getLogger(__name__)


class FeatherlessClient:
    def __init__(self, config):
        self.api_key   = config.api_key
        self.base_url  = config.base_url.rstrip("/")
        self.model     = config.model
        self.timeout   = config.timeout
        self._headers  = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def health_check(self) -> Dict:
        if not self.api_key:
            return {"status": "no_api_key_configured"}
        try:
            r = requests.get(
                f"{self.base_url}/models",
                headers=self._headers,
                timeout=5,
            )
            r.raise_for_status()
            return {"status": "healthy", "models_available": len(r.json().get("data", []))}
        except Exception as e:
            return {"status": f"unreachable: {e}"}

    def complete(self, prompt: str, max_tokens: int = 256, temperature: float = 0.3) -> str:
        """
        Call the Featherless /chat/completions endpoint.
        Returns the assistant message text.
        """
        if not self.api_key:
            raise ValueError("FEATHERLESS_API_KEY not set. Add it to your .env file.")

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers,
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
