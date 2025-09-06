"""Cli Function"""

from pathlib import Path
from pprint import pprint

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

from marqetsim.agent import (
    create_joe_the_analyst,
    create_person,
    generate_coherent_person,
)
from marqetsim.utils import LogCreator, common
from marqetsim.utils.extractor import extract_results_from_agent


@click.group()
def cli():
    """Marq CLI tool."""
    return None


@cli.command()
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
            result = extract_results_from_agent(
                person,
                situation=situation,
                fields=["ad_number", "ad_title"],
                verbose=False,
                logger=logger,
            )
            pprint(result)

        # Save all responses to a json file in the same directory as the input file and with the same name plus "_responses.json"
        output_file_path = Path(file_path).with_name(
            Path(file_path).stem + "_responses.json"
        )
        common.save_json_file(all_response, output_file_path)
        click.echo(f"Responses saved to {output_file_path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
def summarize(file_path):
    """summarize a csv file of agents responses"""

    # read the csv file calculate the count of rows group by response column and show simple visualization through cli
    df = pd.read_csv(file_path)
    counts = df["response"].value_counts()

    console = Console()
    table = Table(title="Response Summary")

    table.add_column("Response", justify="left")
    table.add_column("Count", justify="right")
    table.add_column("Bar", justify="left")

    max_count = counts.max()
    for label, count in counts.items():
        bar_len = int((count / max_count) * 40)  # scale to 40 chars
        bar_text = "█" * bar_len
        table.add_row(str(label), str(count), bar_text)  # convert everything to str

    console.print(table)
