"""
Command-line interface for SoilSpec-PINN.

Provides CLI commands for common operations like preprocessing,
training, and prediction.
"""

import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="soilspec",
    help="SoilSpec-PINN: Physics-Informed Neural Networks for Soil Spectroscopy",
    add_completion=False,
)

console = Console()


@app.command()
def version():
    """Display version information."""
    from soilspec_pinn import __version__

    console.print(f"[bold green]SoilSpec-PINN[/bold green] version {__version__}")


@app.command()
def preprocess(
    input_dir: Path = typer.Argument(..., help="Directory containing OPUS files"),
    output_file: Path = typer.Argument(..., help="Output file for preprocessed data"),
    snv: bool = typer.Option(True, help="Apply SNV correction"),
    derivative: int = typer.Option(0, help="Derivative order (0, 1, or 2)"),
):
    """
    Preprocess spectral data from Bruker OPUS files.

    Example:
        soilspec preprocess data/raw/ data/preprocessed.csv --snv --derivative 1
    """
    console.print(f"[bold blue]Preprocessing spectra from:[/bold blue] {input_dir}")
    console.print(f"[bold blue]Output file:[/bold blue] {output_file}")

    # TODO: Implement preprocessing pipeline
    console.print("[yellow]Preprocessing functionality will be implemented[/yellow]")


@app.command()
def train(
    data_file: Path = typer.Argument(..., help="Path to training data"),
    model_type: str = typer.Option("pinn", help="Model type (pinn, mpnn, pls)"),
    output_dir: Path = typer.Option("models/", help="Directory to save trained model"),
    epochs: int = typer.Option(100, help="Number of training epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
    learning_rate: float = typer.Option(1e-3, help="Learning rate"),
):
    """
    Train a spectral prediction model.

    Example:
        soilspec train data/train.csv --model-type pinn --epochs 100
    """
    console.print(f"[bold blue]Training {model_type} model[/bold blue]")
    console.print(f"Data: {data_file}")
    console.print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")

    # TODO: Implement training pipeline
    console.print("[yellow]Training functionality will be implemented[/yellow]")


@app.command()
def predict(
    model_path: Path = typer.Argument(..., help="Path to trained model"),
    input_file: Path = typer.Argument(..., help="Input spectra file"),
    output_file: Path = typer.Argument(..., help="Output predictions file"),
    uncertainty: bool = typer.Option(False, help="Include uncertainty estimates"),
):
    """
    Make predictions using a trained model.

    Example:
        soilspec predict models/best_model.pt data/test.csv predictions.csv --uncertainty
    """
    console.print(f"[bold blue]Making predictions[/bold blue]")
    console.print(f"Model: {model_path}")
    console.print(f"Input: {input_file}")
    console.print(f"Output: {output_file}")

    # TODO: Implement prediction pipeline
    console.print("[yellow]Prediction functionality will be implemented[/yellow]")


@app.command()
def list_models():
    """
    List available pre-trained models.

    Example:
        soilspec list-models
    """
    console.print("[bold blue]Available Pre-trained Models[/bold blue]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model Name")
    table.add_column("Type")
    table.add_column("Spectral Range")
    table.add_column("Properties")

    # TODO: Query actual available models
    table.add_row("ossl.mir.cubist", "Cubist", "MIR (600-4000 cm⁻¹)", "SOC, Clay, Sand, pH")
    table.add_row("pinn.mir.v1", "PINN", "MIR (600-4000 cm⁻¹)", "SOC, N, P, K")

    console.print(table)
    console.print("\n[yellow]Model listing functionality will be fully implemented[/yellow]")


if __name__ == "__main__":
    app()
