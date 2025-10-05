"""Marqetsim central configuration."""

import configparser
from pathlib import Path

from dotenv import load_dotenv


def activate_dotenv(logger):
    """Load dotenv."""

    env_local = Path.cwd() / ".env.local"
    env_prod = Path.cwd() / ".env"

    if env_local.exists():
        load_dotenv(env_local)
        logger.debug(f"Using {env_local}")
    else:
        load_dotenv(env_prod)
        logger.debug(f"Using {env_prod}")

    if all([env_local.exists(), env_prod.exists()]) is False:
        logger.debug("No .env file found. Will use only system environment variables.")


def read_config_file() -> configparser.ConfigParser:
    """read config file."""

    config = configparser.ConfigParser()

    # Read the default values in the module directory.
    config_file_path = Path(__file__).parent / "config.ini"
    print(f"Looking for default config on: {config_file_path}")

    if config_file_path.exists():
        config.read(config_file_path)
    else:
        raise ValueError(f"Failed to find default config on: {config_file_path}")

    # Now, let's override any specific default value, if there's a custom .ini config.
    # Try the directory of the current main program
    config_file_path = Path.cwd() / "config.ini"
    if config_file_path.exists():

        print(f"Found custom config on: {config_file_path}")

        # override default config
        config.read(config_file_path)
        return config
    else:
        print(f"Failed to find custom config on: {config_file_path}")
        print(
            "Will use only default values. IF THINGS FAIL, TRY CUSTOMIZING MODEL, API TYPE, etc."
        )

    return config
