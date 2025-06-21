import click
import json
from pathlib import Path

from marqetsim.examples import create_joe_the_analyst
from marqetsim.simulator import create_person
from marqetsim.utils.common import read_yaml_file

@click.command()
@click.argument("file_path", type=click.Path(exists=True))
def launch(file_path):
    """Read a YAML or JSON file and print it as a dictionary."""
    try:
        data = read_yaml_file(file_path)
        situation = data.pop("situation")
        options = data.pop("options")
        questions = data.pop("questions")
        if 'agent' not in data:
            people = [create_joe_the_analyst()]
        else:
            one_person = create_person(profile=data.pop("agent"))
            people = [one_person]

        ## TODO: need function to handle path image options
        options_merged = [
            f"#option-{i+1} " + opt.pop("content") for i, opt in enumerate(options)
        ]
        request_msg = f"{questions}\n" + "\n\n".join(options_merged)
        all_response = {}
        for person in people:
            person.set_context(situation)
            all_response[person.name] = person.listen_and_act(request_msg)

        click.echo(data.pop("project"))

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
