"""CLI Function"""

import time
from pathlib import Path
from typing import Any, Dict, List, Union

import click
import pandas as pd
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from marqetsim.agent import (
    create_joe_the_analyst,
    create_person,
    generate_coherent_person,
)
from marqetsim.config import read_config_file, activate_dotenv
from marqetsim.utils import LogCreator, common
from marqetsim.utils.extractor import extract_results_from_agent

console = Console()


def create_agents_from_dict(
    agent_data: Dict[str, Any], settings, logger: LogCreator
) -> List[Any]:
    """Create agents from dictionary configuration."""
    return [create_person(profile=agent_data, settings=settings, logger=logger)]


def create_agents_from_file(
    agent_file_path: str, settings, logger: LogCreator
) -> List[Any]:
    """Create agents from CSV file."""
    file_path = Path(agent_file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Agent file does not exist: {agent_file_path}")

    data_agent = common.read_csv(file_path)
    return [
        create_person(profile=profile, settings=settings, logger=logger)
        for profile in data_agent
    ]


def create_random_agents(count: int, settings, logger: LogCreator) -> List[Any]:
    """Create random agents."""
    people = []
    for _ in range(count):
        record = generate_coherent_person()
        person = create_person(record, settings, logger=logger)
        people.append(person)
    return people


def create_agents(
    agent_config: Union[Dict, str, int, None], settings, logger: LogCreator
) -> List[Any]:
    """Create agents based on configuration."""
    if agent_config is None:
        print("agent is not defined, use predefined agent Joe the Analyst")
        return [create_joe_the_analyst(logger=logger)]

    if isinstance(agent_config, dict):
        return create_agents_from_dict(agent_config, settings, logger)
    elif isinstance(agent_config, str):
        return create_agents_from_file(agent_config, settings, logger)
    elif isinstance(agent_config, int):
        return create_random_agents(agent_config, settings, logger)
    else:
        raise ValueError(
            "Invalid agent definition. Must be a dict, file path, or integer."
        )


def process_options(options: List[Dict[str, Any]]) -> List[str]:
    """Process options into formatted strings."""
    return [f"#option-{i+1} {opt.pop('content')}" for i, opt in enumerate(options)]


def save_responses(responses: Dict[str, Any], input_file_path: str) -> str:
    """Save responses to JSON file."""
    output_file_path = Path(input_file_path).with_name(
        Path(input_file_path).stem + "_responses.json"
    )
    common.save_json_file(responses, output_file_path)
    return str(output_file_path)


def welcome():
    """Welcome message."""

    welcome_path = Path(__file__).parent / "static" / "welcome.txt"
    with open(welcome_path, mode="r", encoding="utf-8") as f:
        welcome_ascii = f.read()

    # Create welcome box content
    welcome_content = f"{welcome_ascii}\n"
    welcome_content += "[bold green]MarqetSim: Multi-Agents LLM Market Research Simulation - CLI[/bold green]\n\n"
    welcome_content += "[dim]Built by [AI Team research](https://github.com/andreaschandra/MarqetSim)[/dim]"

    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="Welcome to MarqetSim",
        subtitle="Multi-Agents LLM Market Research Simulation",
    )
    console.print(Align.center(welcome_box))
    console.print()


def summary(stats):
    """Summary message."""
    table = Table(title="Report Summary")
    table.add_column("# of Agents", justify="center")
    table.add_column("Duration", justify="center")
    table.add_column("API Calls", justify="center")

    table.add_row(
        str(stats["num_agents"]), str(stats["duration"]), str(stats["api_calls"])
    )
    console.print(table)


@click.group()
def cli():
    """Marq CLI tool."""
    pass


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
def launch(file_path: str) -> None:
    """Read a YAML or JSON file and execute agent interactions."""
    start = time.time()

    settings = read_config_file()
    logger = LogCreator(
        log_file="marqetsim_cli.log",
        level=settings["Logging"]["LOGLEVEL"],
    )
    activate_dotenv(logger)

    welcome()

    try:
        data = common.read_yaml_file(file_path)

        # Validate required fields
        required_fields = ["situation", "options", "questions"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        situation = data.pop("situation")
        options = data.pop("options")
        questions = data.pop("questions")

        # Create agents
        agent_config = data.get("agent")
        people = create_agents(agent_config, settings, logger)

        # Process options and create request message
        options_merged = process_options(options)
        request_msg = f"{questions}\n" + "\n\n".join(options_merged)

        # Process each agent
        all_responses = {}
        for person in people:
            person.set_context(situation)
            all_responses[person.name] = {}
            all_responses[person.name]["interaction"] = person.listen_and_act(
                request_msg
            )
            all_responses[person.name]["profile"] = person.get_persona()

            result = extract_results_from_agent(
                person,
                situation=situation,
                fields=["ad_number", "ad_title"],
                verbose=False,
                settings=settings,
                logger=logger,
            )
            all_responses[person.name]["result"] = result

        # Save responses
        output_path = save_responses(all_responses, file_path)

        stats = {}
        stats["num_agents"] = len(people)
        stats["duration"] = round(time.time() - start, 2)
        stats["api_calls"] = sum(person.api_call_count for person in people)
        summary(stats)

        click.echo(f"Responses saved to {output_path}")
    except FileNotFoundError as e:
        click.echo(f"File error: {e}", err=True)
    except ValueError as e:
        click.echo(f"Configuration error: {e}", err=True)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
def summarize(file_path: str) -> None:
    """Summarize a CSV file of agent responses."""
    try:
        df = pd.read_csv(file_path)

        if "response" not in df.columns:
            click.echo("Error: CSV file must contain a 'response' column", err=True)
            return

        counts = df["response"].value_counts()

        if counts.empty:
            click.echo("No data found in response column")
            return

        console = Console()
        table = Table(title="Response Summary")

        table.add_column("Response", justify="left")
        table.add_column("Count", justify="right")
        table.add_column("Bar", justify="left")

        max_count = counts.max()
        for label, count in counts.items():
            bar_len = int((count / max_count) * 40)
            bar_text = "█" * bar_len
            table.add_row(str(label), str(count), bar_text)

        console.print(table)
    except pd.errors.EmptyDataError:
        click.echo("Error: CSV file is empty", err=True)
    except pd.errors.ParserError as e:
        click.echo(f"Error parsing CSV file: {e}", err=True)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
