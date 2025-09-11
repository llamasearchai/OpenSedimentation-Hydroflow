"""Report generator tests."""

from pathlib import Path
from hydroflow.utils.reporting import ReportGenerator


def test_report_html_tmp(tmp_path: Path):
    gen = ReportGenerator({})
    data = {"metric_a": 1.23, "metric_b": 4.56}
    out = tmp_path / "report.html"
    gen.create_report(data, "bathymetry", "html", str(out))
    text = out.read_text(encoding="utf-8")
    assert "HydroFlow Report" in text
    assert "metric_a" in text and "1.23" in text

