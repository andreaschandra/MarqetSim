import argparse
import json

from ollama import Client
from pydantic import BaseModel

from tinytroupe import utils


class OllamaAPIClient:
    def __init__(self):
        self.default = None

        self.get_config()
        self.client = Client(self.default["URL"])

    def get_config(self):

        config = utils.read_config_file()

        default = {}
        default["URL"] = config["Ollama"].get("URL", "localhost:11434")
        default["MODEL"] = config["Ollama"].get("MODEL", "tinyllama")

        self.default = default

    def send_message(self, current_message=None, response_format=None):
        """
        Sends a POST request with the given data to the specified URL and streams the response.

        Args:
            request_data (dict): The data to be sent as the request body.
            endpoint_url (str): The endpoint URL.

        Returns:
            list: A list of decoded JSON objects from the response.
        """

        messages = current_message
        model = self.default["MODEL"]
        format = response_format.model_json_schema()

        response = self.client.chat(messages=messages, model=model, format=format)
        print(f"Raw response: {response}")

        return {"role": response.message.role, "content": response.message.content}


class Country(BaseModel):
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

    ollama = OllamaAPIClient()

    response = ollama.send_message(current_message, response_format)
    response_dict = json.loads(response.message.content)
    print("Question: ", args.question)
    print("Country: ", response_dict["name"])
    print("Capital: ", response_dict["capital"])
    print("Languages: ", response_dict["languages"])


if __name__ == "__main__":
    main()
