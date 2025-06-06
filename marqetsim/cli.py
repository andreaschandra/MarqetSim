import click
import json
import yaml
from pathlib import Path

from marqetsim.examples import create_joe_the_analyst


def read_file(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File '{file_path}' not found.")

    with open(file_path, "r") as f:
        if file_path.endswith(".json"):
            return json.load(f)
        elif file_path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        else:
            raise ValueError("Unsupported file format. Use .json or .yaml/.yml")


@click.command()
@click.argument("file_path", type=click.Path(exists=True))
def launch(file_path):
    """Read a YAML or JSON file and print it as a dictionary."""
    try:
        data = read_file(file_path)
        situation = data.pop("situation")
        options = data.pop("options")
        questions = data.pop("questions")

        ## need funtion to handle path image options
        options_merged = [
            f"#option-{i+1} " + opt.pop("content") for i, opt in enumerate(options)
        ]
        request_msg = f"{questions}\n" + "\n\n".join(options_merged)
        print("=" * 20 + "create Person")
        people = [create_joe_the_analyst()]
        print("=" * 20 + "Person created")
        all_response = {}
        for person in people:
            print("=" * 20 + "set context")
            person.set_context(situation)
            print("=" * 20 + "context is set")
            all_response[person.name] = person.listen_and_act(request_msg)

        click.echo(data.pop("project"))

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
