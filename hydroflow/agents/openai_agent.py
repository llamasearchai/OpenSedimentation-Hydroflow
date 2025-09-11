"""Simple OpenAI client wrapper for HydroFlow suggestions and summaries."""

from __future__ import annotations

import os
from typing import Dict

from openai import OpenAI


class HydroFlowAgent:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
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


