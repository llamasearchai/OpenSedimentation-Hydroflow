"""Command-line interface for HydroFlow."""

import json
import logging
import sys
from pathlib import Path

import click

from hydroflow import __version__
from hydroflow.agents.openai_agent import HydroFlowAgent
from hydroflow.analysis.bathymetric import BathymetricAnalyzer
from hydroflow.analysis.sediment_transport import SedimentProperties, SedimentTransportModel
from hydroflow.analysis.vegetation import VegetationAnalyzer
from hydroflow.core.config import Config
from hydroflow.utils.reporting import ReportGenerator

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--debug", is_flag=True, help="Debug mode")
@click.pass_context
def cli(ctx, config, verbose, debug):
    """HydroFlow - Advanced Bathymetric Analysis and Sediment Management System."""
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

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
@click.option("--check", is_flag=True, help="Check system dependencies (non-interactive)")
@click.pass_context
def info(ctx, check):
    """Display system information."""
    click.echo(f"HydroFlow Version: {__version__}")
    click.echo(f"Python Version: {sys.version.split()[0]}")
    click.echo(f"Configuration: {ctx.obj.environment}")

    if check:

        def _try_import(name):
            try:
                module = __import__(name)
                version = getattr(module, "__version__", "unknown")
                click.echo(f"OK {name}: {version}")
            except Exception:
                click.echo(f"MISS {name}")

        for dep in ["numpy", "scipy", "pandas", "xarray", "sklearn"]:
            _try_import(dep)


@cli.group()
@click.pass_context
def analyze(ctx):
    """Run analysis operations."""
    pass


@analyze.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file (JSON)")
@click.option(
    "--method", type=click.Choice(["kriging", "idw", "spline", "triangulation"]), default="idw"
)
@click.option("--resolution", type=float, default=1.0, help="Grid resolution")
@click.pass_context
def bathymetry(ctx, input_file, output, method, resolution):
    """Analyze bathymetric data from a whitespace-delimited XYZ file."""
    import numpy as np

    analyzer = BathymetricAnalyzer(ctx.obj.dict())
    data = np.loadtxt(input_file)
    surface = analyzer.create_bathymetric_surface(data[:, :3], method=method, resolution=resolution)
    features = analyzer.detect_channel_features(surface)
    result = {"shape": list(surface.shape), "metrics": features["metrics"]}
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        click.echo(f"Saved {output}")
    else:
        click.echo(json.dumps(result, indent=2))


@analyze.command()
@click.argument("flow_csv", type=click.Path(exists=True))
@click.option("--d50", type=float, default=0.5, help="Median grain size (mm)")
@click.option("--d90", type=float, default=1.0, help="90th percentile grain size (mm)")
@click.option("--output", "-o", type=click.Path(), help="Output file (JSON)")
@click.pass_context
def sediment(ctx, flow_csv, d50, d90, output):
    """Analyze sediment transport from CSV with columns velocity,depth."""
    import numpy as np
    import pandas as pd

    df = pd.read_csv(flow_csv)
    v = df["velocity"].to_numpy()
    h = df["depth"].to_numpy()
    model = SedimentTransportModel(ctx.obj.dict())
    model.set_sediment_properties(SedimentProperties(d50=d50, d90=d90))
    tau = model.calculate_bed_shear_stress(v, h)
    bedload = model.calculate_bedload_transport(tau)
    suspended = model.calculate_suspended_load(v, h)
    result = {
        "mean_bedload": float(np.mean(bedload)),
        "max_bedload": float(np.max(bedload)),
        "mean_suspended": float(np.mean(suspended)),
        "total_transport": float(np.sum(bedload + suspended)),
    }
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        click.echo(f"Saved {output}")
    else:
        click.echo(json.dumps(result, indent=2))


@analyze.command()
@click.argument("imagery_path", type=click.Path(exists=True))
@click.option(
    "--bands",
    multiple=True,
    default=["red", "green", "blue", "nir"],
    help="Band names order for .npy data",
)
@click.pass_context
def vegetation(ctx, imagery_path, bands):
    """Analyze vegetation from .npy imagery (shape: bands x H x W)."""
    import numpy as np

    analyzer = VegetationAnalyzer(ctx.obj.dict())
    arr = np.load(imagery_path)
    indices = analyzer._calculate_vegetation_indices(arr, list(bands))
    classification = analyzer._classify_vegetation(arr, indices)
    metrics = analyzer._calculate_vegetation_metrics(classification)
    click.echo(json.dumps(metrics, indent=2))


@cli.group()
@click.pass_context
def api(ctx):
    """API operations."""
    pass


@api.command("serve")
@click.option("--host", default="0.0.0.0", help="Host")
@click.option("--port", type=int, default=8000, help="Port")
@click.pass_context
def serve_api(ctx, host, port):
    """Serve FastAPI application using Uvicorn."""
    import uvicorn

    uvicorn.run("hydroflow.api.app:app", host=host, port=port, reload=False)


@cli.group()
@click.pass_context
def agents(ctx):
    """OpenAI agent operations."""
    pass


@agents.command("summarize")
@click.option(
    "--metrics", "-m", type=click.Path(exists=True), required=True, help="JSON file with metrics"
)
def agents_summarize(metrics):
    """Summarize metrics using the OpenAI agent."""
    import json as _json
    import os

    if not os.getenv("OPENAI_API_KEY"):
        click.echo("OPENAI_API_KEY not configured", err=True)
        sys.exit(2)
    with open(metrics, "r") as f:
        m = _json.load(f)
    agent = HydroFlowAgent()
    summary = agent.summarize_metrics(m)
    click.echo(summary)


if __name__ == "__main__":
    cli()


@cli.command()
@click.option("--input", "-i", type=click.Path(exists=True), required=True, help="Input JSON data")
@click.option(
    "--type", "type_", type=click.Choice(["bathymetry", "flow", "sediment"]), required=True
)
@click.option("--format", "format_", type=click.Choice(["html"]), default="html")
@click.option("--output", "-o", type=click.Path(), required=True, help="Output report file")
@click.pass_context
def report(ctx, input, type_, format_, output):
    """Generate analysis report."""
    with open(input, "r") as f:
        data = json.load(f)
    generator = ReportGenerator(ctx.obj.dict())
    generator.create_report(data, type_, format_, output)
    click.echo(f"Report saved to {output}")
