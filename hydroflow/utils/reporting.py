"""Simple report generator for HydroFlow."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


class ReportGenerator:
    def __init__(self, config: Dict):
        self.config = config

    def create_report(self, data: Dict, report_type: str, fmt: str, output: str) -> None:
        fmt = fmt.lower()
        if fmt not in {"html"}:
            # For simplicity, support HTML only in this minimal implementation
            fmt = "html"
        html = self._render_html(data, report_type)
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")

    def _render_html(self, data: Dict, report_type: str) -> str:
        title = f"HydroFlow Report - {report_type.title()}"
        body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in data.items())
        return (
            "<html><head><meta charset='utf-8'><title>" + title + "</title></head>"
            "<body><h1>" + title + "</h1>"
            "<table border='1' cellpadding='6' cellspacing='0'>" + body + "</table>"
            "</body></html>"
        )
