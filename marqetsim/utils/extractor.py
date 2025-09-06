"""Answer extractor to structured form."""

from pathlib import Path

import chevron

from marqetsim.llm.anthropic import AnthropicAPIClient
from marqetsim.utils import common


def extract_results_from_agents(
    agents,
    extraction_objective: str = None,
    situation: str = None,
    fields: list = None,
    fields_hints: dict = None,
    verbose: bool = None,
    logger=None,
):
    """
    Extracts results from a list of TinyPerson instances.

    Args:
        agents (List[TinyPerson]): The list of TinyPerson instances to extract results from.
        extraction_objective (str): The extraction objective.
        situation (str): The situation to consider.
        fields (list, optional): The fields to extract. If None, the extractor will decide what names to use.
            Defaults to None.
        fields_hints (dict, optional): Hints for the fields to extract. Maps field names to strings with the hints. Defaults to None.
        verbose (bool, optional): Whether to print debug messages. Defaults to False.


    """
    results = []
    for agent in agents:
        result = extract_results_from_agent(
            agent,
            extraction_objective,
            situation,
            fields,
            fields_hints,
            verbose,
            logger,
        )
        results.append(result)

    return results


def extract_results_from_agent(
    agent,
    extraction_objective: str = "The main points present in the agent's interactions history.",
    situation: str = "",
    fields: list = None,
    fields_hints: dict = None,
    verbose: bool = None,
    logger=None,
):
    """
    Extracts results from a TinyPerson instance.

    Args:
        tinyperson (TinyPerson): The TinyPerson instance to extract results from.
        extraction_objective (str): The extraction objective.
        situation (str): The situation to consider.
        fields (list, optional): The fields to extract. If None, the extractor will decide what names to use.
            Defaults to None.
        fields_hints (dict, optional): Hints for the fields to extract. Maps field names to strings with the hints. Defaults to None.
        verbose (bool, optional): Whether to print debug messages. Defaults to False.
    """

    client = AnthropicAPIClient()

    extractor_prompt_path = (
        Path(__file__).parent.parent
        / "prompts"
        / "interaction_results_extractor.mustache"
    )

    messages = []

    rendering_configs = {}
    if fields is not None:
        rendering_configs["fields"] = ", ".join(fields)

    if fields_hints is not None:
        rendering_configs["fields_hints"] = list(fields_hints.items())

    with open(extractor_prompt_path, "r", encoding="utf-8", errors="replace") as f:
        extract_prompt_content = f.read()

    content = chevron.render(extract_prompt_content, rendering_configs)

    interaction_history = agent.episodic_memory.retrieve()

    extraction_request_prompt = f"""
        ## Extraction objective

        {extraction_objective}

        ## Situation
        You are considering a single agent, named {agent.name}. Your objective thus refers to this agent specifically.
        {situation}

        ## Agent Interactions History

        You will consider an agent's history of interactions, which include stimuli it received as well as actions it 
        performed.

        {interaction_history}
        """
    messages.append({"role": "user", "content": extraction_request_prompt})

    next_message = client.send_message(messages=messages, system_message=content)

    debug_msg = f"Extraction raw result message: {next_message}"
    logger.debug(debug_msg)
    if verbose:
        print(debug_msg)

    if (next_message is not None) & isinstance(next_message, str):
        result = common.extract_json(next_message["content"])
    elif isinstance(next_message, dict):
        result = next_message.get("content", None)
    else:
        result = None

    return result
