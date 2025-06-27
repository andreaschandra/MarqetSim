"""Cli Function"""

from pathlib import Path

import click

from marqetsim.examples import create_joe_the_analyst
from marqetsim.simulator import create_person
from marqetsim.utils import common


@click.command()
@click.argument("file_path", type=click.Path(exists=True))
def launch(file_path):
    """Read a YAML or JSON file and print it as a dictionary."""
    try:
        data = common.read_yaml_file(file_path)
        situation = data.pop("situation")
        options = data.pop("options")
        questions = data.pop("questions")

        if "agent" not in data:
            print("agent is not defined, use predefined agent Joe the Analyst")
            people = [create_joe_the_analyst()]
        else:
            if isinstance(data["agent"], dict):
                one_person = create_person(profile=data.pop("agent"))
                people = [one_person]
            elif isinstance(data["agent"], str):
                agent_file_path = Path(data["agent"])
                assert agent_file_path.is_file(), "Agent file does not exist."
                data_agent = common.read_csv(agent_file_path)
                people = [create_person(profile=profile) for profile in data_agent]
            else:
                raise ValueError("Invalid agent definition. Must be a dict or a file path.")

        # TODO: need function to handle path image options
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
