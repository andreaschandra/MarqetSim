import click
import json
import yaml
from pathlib import Path


def read_file(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File '{file_path}' not found.")

    with open(file_path, 'r') as f:
        if file_path.endswith('.json'):
            return json.load(f)
        elif file_path.endswith(('.yaml', '.yml')):
            return yaml.safe_load(f)
        else:
            raise ValueError("Unsupported file format. Use .json or .yaml/.yml")


@click.command()
@click.argument('file_path', type=click.Path(exists=True))
def cli(file_path):
    """Read a YAML or JSON file and print it as a dictionary."""
    try:
        data = read_file(file_path)
        click.echo(data)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
