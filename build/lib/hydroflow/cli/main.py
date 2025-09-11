"""Command-line interface for HydroFlow."""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from hydroflow import __version__
from hydroflow.core.config import Config


logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--debug', is_flag=True, help='Debug mode')
@click.pass_context
def cli(ctx, config, verbose, debug):
    """HydroFlow - Advanced Bathymetric Analysis and Sediment Management System."""
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if config:
        ctx.obj = Config.from_yaml(Path(config))
    else:
        ctx.obj = Config()

    logger.info(f"HydroFlow v{__version__} initialized")


@cli.command()
@click.pass_context
def config(ctx):
    """Display current configuration."""
    click.echo(json.dumps(ctx.obj.dict(), indent=2))


@cli.command()
@click.option('--check', is_flag=True, help='Check system dependencies (non-interactive)')
@click.pass_context
def info(ctx, check):
    """Display system information."""
    click.echo(f"HydroFlow Version: {__version__}")
    click.echo(f"Python Version: {sys.version.split()[0]}")
    click.echo(f"Configuration: {ctx.obj.environment}")

    if check:
        # Non-interactive dependency check
        def _try_import(name):
            try:
                module = __import__(name)
                version = getattr(module, '__version__', 'unknown')
                click.echo(f"OK {name}: {version}")
            except Exception:
                click.echo(f"MISS {name}")

        for dep in ["numpy", "scipy", "pandas", "xarray", "sklearn"]:
            _try_import(dep)


if __name__ == "__main__":
    cli()


