"""Smoke tests for CLI."""

from click.testing import CliRunner

from hydroflow.cli.main import cli


def test_cli_info_noninteractive():
    runner = CliRunner()
    result = runner.invoke(cli, ["info", "--check"])  # non-interactive
    assert result.exit_code == 0
    assert "HydroFlow Version" in result.output


def test_cli_config():
    runner = CliRunner()
    result = runner.invoke(cli, ["config"])  # prints JSON
    assert result.exit_code == 0
    assert "environment" in result.output
