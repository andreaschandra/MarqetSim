import sys
from click.testing import CliRunner
from marqetsim.cli import launch  # Adjust this import as needed
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python debug_cli.py <path-to-arg1>")
    sys.exit(1)

arg1_path = sys.argv[1]

# Optional: validate the path
if not Path(arg1_path).exists():
    print(f"Error: path '{arg1_path}' does not exist.")
    sys.exit(1)

# Run the CLI
runner = CliRunner()
result = runner.invoke(launch, [arg1_path])

print("OUTPUT:")
print(result.output)

if result.exception:
    raise result.exception
