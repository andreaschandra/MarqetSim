"""Claude client."""

import json
import logging
import os
import re

from anthropic import Anthropic
from pydantic import BaseModel

logger = logging.getLogger("marqetsim")


class AnthropicAPIClient:
    """Claude client."""

    def __init__(self):

        self.client = Anthropic(
            api_key=os.environ.get(
                "ANTHROPIC_API_KEY"
            ),  # This is the default and can be omitted
        )
        self.response = None

    def send_message(self, message, response_format):
        """
        Simulate sending a message to Claude and receiving a response.
        In a real implementation, this would interact with the Claude API.
        """

        response_format = response_format.model_json_schema()

        raw_response = self.client.messages.create(
            max_tokens=1024,
            messages=message,
            model="claude-3-5-haiku-latest",
        )

        logger.debug(f"Raw response: {raw_response}\n\n")
        text = raw_response.content[0].text
        logger.debug(f"content.text: {text}\n\n")

        text = re.sub(r"\`\`\`json\n|\`\`\`", "", text)

        try:
            response_json = [json.loads(text)]
        except Exception as e:
            try:
                list_of_response = text.split("\n\n")
                if list_of_response[0].startswith("{"):
                    response_json = [json.loads(res) for res in list_of_response]
                else:
                    response_json = list_of_response[1:]
                    response_json = [json.loads(res) for res in response_json]
            except Exception as e:
                logger.error(f"Error parsing response: {e} \n Text: {text}")

        self.response = {"role": raw_response.role, "content": response_json}

        return self.response

    def placeholder(self):
        """Placeholder response."""
        data_type = self.response
        return data_type


if __name__ == "__main__":
    # Example usage

    class ResponseFormat(BaseModel):
        """Response format for the weather query."""

        date: str
        condition: str

    client = AnthropicAPIClient()
    response = client.send_message(
        "What is the weather like today in Jakarta?", response_format=ResponseFormat
    )
    print(f"Response text: {response}")
    print(f"Date: {response['content']['date']}")
    print(f"Condition: {response['content']['condition']}")
