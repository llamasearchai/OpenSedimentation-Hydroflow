"""Simple OpenAI client wrapper for HydroFlow suggestions and summaries.

This module avoids importing the OpenAI SDK at module import time so that
environments without the dependency (e.g., CI) can still import HydroFlow.
The SDK is only imported when an instance is created and used.
"""

from __future__ import annotations

import os
from typing import Dict


class HydroFlowAgent:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        # Lazy import to avoid hard dependency during module import
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover - environment specific
            raise RuntimeError("OpenAI SDK not available. Install 'openai' to use agents.") from e
        self.client = OpenAI(api_key=api_key)

    def summarize_metrics(self, metrics: Dict) -> str:
        prompt = (
            "Summarize the following hydrological metrics for a technical audience in <= 120 words.\n"
            + str(metrics)
        )
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
