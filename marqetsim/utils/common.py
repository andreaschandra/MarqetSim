"""Agent utilities"""

import configparser
import copy
import csv
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Collection, Union

import yaml
from dotenv import load_dotenv

# logger
logger = logging.getLogger("marqetsim")


def read_yaml_file(file_path):
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


def extract_json(text: str) -> dict:
    """
    Extracts a JSON object from a string, ignoring: any text before the first
    opening curly brace; and any Markdown opening (```json) or closing(```) tags.
    """
    try:
        # remove any text before the first opening curly or square braces, using regex. Leave the braces.
        text = re.sub(r"^.*?({|\[)", r"\1", text, flags=re.DOTALL)

        # remove any trailing text after the LAST closing curly or square braces, using regex. Leave the braces.
        text = re.sub(r"(}|\])(?!.*(\]|\})).*$", r"\1", text, flags=re.DOTALL)

        # remove invalid escape sequences, which show up sometimes
        # replace \' with just '
        text = re.sub("\\'", "'", text)  # re.sub(r'\\\'', r"'", text)

        # remove new lines, tabs, etc.
        text = text.replace("\n", "").replace("\t", "").replace("\r", "")

        # return the parsed JSON object
        return json.loads(text)

    except Exception:
        return {}


def truncate_actions_or_stimuli(
    list_of_actions_or_stimuli: Collection[dict], max_content_length: int
) -> Collection[str]:
    """
    Truncates the content of actions or stimuli at the specified maximum length. Does not modify the original list.

    Args:
        list_of_actions_or_stimuli (Collection[dict]): The list of actions or stimuli to truncate.
        max_content_length (int): The maximum length of the content.

    Returns:
        Collection[str]: The truncated list of actions or stimuli. It is a new list, not a reference to the original list,
        to avoid unexpected side effects.
    """
    cloned_list = copy.deepcopy(list_of_actions_or_stimuli)

    for element in cloned_list:
        # the external wrapper of the LLM message: {'role': ..., 'content': ...}
        if "content" in element:
            msg_content = element["content"]

            # now the actual action or stimulus content

            # has action, stimuli or stimulus as key?
            if "action" in msg_content:
                # is content there?
                if "content" in msg_content["action"]:
                    msg_content["action"]["content"] = break_text_at_length(
                        msg_content["action"]["content"], max_content_length
                    )

            elif "stimulus" in msg_content:
                # is content there?
                if "content" in msg_content["stimulus"]:
                    msg_content["stimulus"]["content"] = break_text_at_length(
                        msg_content["stimulus"]["content"], max_content_length
                    )

            elif "stimuli" in msg_content:
                # for each element in the list
                for stimulus in msg_content["stimuli"]:
                    # is content there?
                    if "content" in stimulus:
                        stimulus["content"] = break_text_at_length(
                            stimulus["content"], max_content_length
                        )

    return cloned_list


def break_text_at_length(text: Union[str, dict], max_length: int = None) -> str:
    """
    Breaks the text (or JSON) at the specified length, inserting a "(...)" string at the break point.
    If the maximum length is `None`, the content is returned as is.
    """
    if isinstance(text, dict):
        text = json.dumps(text, indent=4)

    if max_length is None or len(text) <= max_length:
        return text
    else:
        return text[:max_length] + " (...)"


def repeat_on_error(retries: int, exceptions: list):
    """
    Decorator that repeats the specified function call if an exception among those specified occurs,
    up to the specified number of retries. If that number of retries is exceeded, the
    exception is raised. If no exception occurs, the function returns normally.

    Args:
        retries (int): The number of retries to attempt.
        exceptions (list): The list of exception classes to catch.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except tuple(exceptions) as e:
                    logger.debug(f"Exception occurred: {e}")
                    if i == retries - 1:
                        raise e
                    else:
                        logger.debug(f"Retrying ({i+1}/{retries})...")
                        continue

        return wrapper

    return decorator


_CONFIG = None


def read_config_file(use_cache=True, verbose=True) -> configparser.ConfigParser:
    """read config file."""
    global _CONFIG
    if use_cache and _CONFIG is not None:
        # if we have a cached config and accept that, return it
        return _CONFIG

    else:
        config = configparser.ConfigParser()

        # Read the default values in the module directory.
        config_file_path = (
            Path(__file__).parent.parent.absolute() / "settings" / "config.ini"
        )
        if verbose:
            print(f"Looking for default config on: {config_file_path}")

        if config_file_path.exists():
            config.read(config_file_path)
            _CONFIG = config
        else:
            raise ValueError(f"Failed to find default config on: {config_file_path}")

        # Now, let's override any specific default value, if there's a custom .ini config.
        # Try the directory of the current main program
        config_file_path = Path.cwd() / "config.ini"
        if config_file_path.exists():

            if verbose:
                print(f"Found custom config on: {config_file_path}")

            config.read(
                config_file_path
            )  # this only overrides the values that are present in the custom config
            _CONFIG = config
            return config
        else:
            if verbose:
                logger.warning(f"Failed to find custom config on: {config_file_path}")
                logger.warning(
                    "Will use only default values. IF THINGS FAIL, TRY CUSTOMIZING MODEL, API TYPE, etc."
                )

        return config


config = read_config_file()


def add_rai_template_variables_if_enabled(template_variables: dict) -> dict:
    """
    Adds the RAI template variables to the specified dictionary, if the RAI disclaimers are enabled.
    These can be configured in the config.ini file. If enabled, the variables will then load the RAI disclaimers from the
    appropriate files in the prompts directory. Otherwise, the variables will be set to None.

    Args:
        template_variables (dict): The dictionary of template variables to add the RAI variables to.

    Returns:
        dict: The updated dictionary of template variables.
    """

    rai_harmful_content_prevention = config["Simulation"].getboolean(
        "RAI_HARMFUL_CONTENT_PREVENTION", True
    )
    rai_copyright_infringement_prevention = config["Simulation"].getboolean(
        "RAI_COPYRIGHT_INFRINGEMENT_PREVENTION", True
    )

    # Harmful content
    with open(
        os.path.join(
            Path(__file__).parent.parent.absolute(),
            "prompts/rai_harmful_content_prevention.md",
        ),
        encoding="utf-8",
        mode="r",
    ) as f:
        rai_harmful_content_prevention_content = f.read()

    template_variables["rai_harmful_content_prevention"] = (
        rai_harmful_content_prevention_content
        if rai_harmful_content_prevention
        else None
    )

    # Copyright infringement
    with open(
        os.path.join(
            Path(__file__).parent.parent.absolute(),
            "prompts/rai_copyright_infringement_prevention.md",
        ),
        encoding="utf-8",
        mode="r",
    ) as f:
        rai_copyright_infringement_prevention_content = f.read()

    template_variables["rai_copyright_infringement_prevention"] = (
        rai_copyright_infringement_prevention_content
        if rai_copyright_infringement_prevention
        else None
    )

    return template_variables


def sanitize_raw_string(value: str) -> str:
    """
    Sanitizes the specified string by:
      - removing any invalid characters.
      - ensuring it is not longer than the maximum Python string length.

    This is for an abundance of caution with security, to avoid any potential issues with the string.
    """

    # remove any invalid characters by making sure it is a valid UTF-8 string
    value = value.encode("utf-8", "ignore").decode("utf-8")

    # ensure it is not longer than the maximum Python string length
    return value[: sys.maxsize]


def sanitize_dict(value: dict) -> dict:
    """
    Sanitizes the specified dictionary by:
      - removing any invalid characters.
      - ensuring that the dictionary is not too deeply nested.
    """

    # sanitize the string representation of the dictionary
    tmp_str = sanitize_raw_string(json.dumps(value, ensure_ascii=False))

    value = json.loads(tmp_str)

    # ensure that the dictionary is not too deeply nested
    return value


class Config:
    def __init__(self):
        if os.path.exists(".env.local"):
            load_dotenv(".env.local")
            logger.info("Using .env.local")
        else:
            load_dotenv()
            logger.info("Using .env")


class RichTextStyle:
    """Log Styling."""

    STIMULUS_CONVERSATION_STYLE = "bold italic cyan1"
    STIMULUS_THOUGHT_STYLE = "dim italic cyan1"
    STIMULUS_DEFAULT_STYLE = "italic"
    ACTION_DONE_STYLE = "grey82"
    ACTION_TALK_STYLE = "bold green3"
    ACTION_THINK_STYLE = "green"
    ACTION_DEFAULT_STYLE = "purple"

    @classmethod
    def get_style_for(cls, kind: str, event_type: str):
        """style for kind and event type."""

        if kind == "stimulus" or kind == "stimuli":
            if event_type == "CONVERSATION":
                return cls.STIMULUS_CONVERSATION_STYLE
            elif event_type == "THOUGHT":
                return cls.STIMULUS_THOUGHT_STYLE
            else:
                return cls.STIMULUS_DEFAULT_STYLE

        elif kind == "action":
            if event_type == "DONE":
                return cls.ACTION_DONE_STYLE
            elif event_type == "TALK":
                return cls.ACTION_TALK_STYLE
            elif event_type == "THINK":
                return cls.ACTION_THINK_STYLE
            else:
                return cls.ACTION_DEFAULT_STYLE

def read_csv(path: str) -> list:
    """
    Reads a CSV file and returns a list of dictionaries, where each dictionary represents a row in the CSV file.
    The keys of the dictionary are the column names.
    """

    with open(path, mode="r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        return [row for row in reader]
