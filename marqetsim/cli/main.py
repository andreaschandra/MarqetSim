"""Cli Function"""

from pathlib import Path

import click

from marqetsim.agent import (
    create_joe_the_analyst,
    create_person,
    generate_coherent_person,
)
from marqetsim.utils import LogCreator, common


@click.command()
@click.argument("file_path", type=click.Path(exists=True))
def launch(file_path):
    """Read a YAML or JSON file and print it as a dictionary."""

    logger = LogCreator()
    try:
        data = common.read_yaml_file(file_path)
        situation = data.pop("situation")
        options = data.pop("options")
        questions = data.pop("questions")

        people = []
        if "agent" not in data:
            print("agent is not defined, use predefined agent Joe the Analyst")
            people = [create_joe_the_analyst()]
        else:
            if isinstance(data["agent"], dict):
                one_person = create_person(profile=data.pop("agent"), logger=logger)
                people = [one_person]
            elif isinstance(data["agent"], str):
                agent_file_path = Path(data["agent"])
                assert agent_file_path.is_file(), "Agent file does not exist."
                data_agent = common.read_csv(agent_file_path)
                people = [
                    create_person(profile=profile, logger=logger)
                    for profile in data_agent
                ]
            elif isinstance(data["agent"], int):
                for i in range(data["agent"]):
                    record = generate_coherent_person()
                    person = create_person(record, logger=logger)
                    people.append(person)
            else:
                raise ValueError(
                    "Invalid agent definition. Must be a dict or a file path."
                )

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
