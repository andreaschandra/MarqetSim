"""ollama utility functions"""

import argparse
import json

from ollama import Client
from pydantic import BaseModel

from marqetsim.llm.base import LLMBase
from marqetsim.utils import LogCreator


class OllamaAPIClient(LLMBase):
    """ollama client"""

    def __init__(self, settings, logger=LogCreator()):

        self.settings = settings
        self.client = Client(self.settings["Ollama"]["URL"])
        self.logger = logger
        self.settings = settings

    def send_message(self, messages, system_message, response_format=None):
        """
        Sends a chat message to the LLM client with the specified messages, system prompt, and response format.
        Args:
            messages (list): A list of message dictionaries, each containing 'role' and 'content' keys.
            system_message (str): The system prompt to be included at the beginning of the conversation.
            response_format: An object specifying the expected response format, with a `model_json_schema()` method.
        Returns:
            dict: A dictionary containing the 'role' and 'content' of the response message from the LLM.
        Prints:
            The raw response received from the LLM client for debugging purposes.
        """

        model = self.settings["Ollama"]["MODEL"]

        if response_format:
            response_format = response_format.model_json_schema()

        if system_message:
            message = [{"role": "system", "content": system_message}]
            messages = message + messages

        response = self.client.chat(
            messages=messages, model=model, format=response_format
        )
        print(f"Raw response: {response}")

        return {
            "role": response["message"]["role"],
            "content": response["message"]["content"],
        }


class Country(BaseModel):
    """base abstract class for country model"""

    name: str
    capital: str
    languages: list[str]


def main():
    """Main function to send a question to the API and print the answer."""
    parser = argparse.ArgumentParser(
        description="Send a question to the API and get an answer."
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Tell me about Canada.",
        required=False,
        help="The question to ask the API.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tinyllama",
        help="The model to use for the API request.",
    )
    args = parser.parse_args()

    current_message = [
        {
            "role": "user",
            "content": args.question,
        }
    ]
    response_format = Country

    settings = {}
    settings["Ollama"] = {}
    settings["Ollama"]["URL"] = "localhost:11434"
    settings["Ollama"]["MODEL"] = "tinyllama"
    ollama = OllamaAPIClient(settings)

    response = ollama.send_message(current_message, response_format)
    response_dict = json.loads(response["content"])
    print("Question: ", args.question)
    print("Country: ", response_dict["name"])
    print("Capital: ", response_dict["capital"])
    print("Languages: ", response_dict["languages"])


if __name__ == "__main__":
    main()
