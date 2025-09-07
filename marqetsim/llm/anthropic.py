"""Claude client."""

import json
import os
import re

from anthropic import NOT_GIVEN, Anthropic
from pydantic import BaseModel

from marqetsim.llm.base import LLMBase
from marqetsim.utils import LogCreator


class AnthropicAPIClient(LLMBase):
    """Claude client."""

    def __init__(self, settings, logger=LogCreator()):

        self.client = Anthropic(
            api_key=os.environ.get(
                "ANTHROPIC_API_KEY"
            ),  # This is the default and can be omitted
        )
        self.logger = logger
        self.settings = settings
        self.response = None

    def send_message(
        self,
        messages,
        system_message: str = NOT_GIVEN,
    ):
        """
        Simulate sending a message to Claude and receiving a response.
        In a real implementation, this would interact with the Claude API.
        """

        raw_response = self.client.messages.create(
            system=system_message,
            messages=messages,
            max_tokens=1024,
            model="claude-3-5-haiku-latest",
        )

        self.logger.debug(f"Raw response: {raw_response}\n\n")
        text = raw_response.content[0].text
        self.logger.debug(f"content.text: {text}\n\n")

        text = re.sub(r"\`\`\`json\n|\`\`\`", "", text)

        try:
            response_json = [json.loads(text)]
        except Exception as e:
            try:
                list_of_response = text.split("\n\n")
                if list_of_response[0].startswith("{"):
                    response_json = [
                        json.loads(res, strict=False) for res in list_of_response
                    ]
                else:
                    response_json = list_of_response[1:]
                    response_json = [
                        json.loads(res, strict=False) for res in response_json
                    ]
            except Exception as e:
                self.logger.error(f"Error parsing response: {e} \n Text: {text}")
                self.logger.debug("Sending to llm to fix the json")
                try:
                    message = [
                        {
                            "role": "user",
                            "content": f"Fix this json: \n{text} \nmake sure the json is valid",
                        }
                    ]
                    next_message = self.client.messages.create(
                        messages=message,
                        max_tokens=1024,
                        model="claude-3-5-haiku-latest",
                    )
                    response_json = json.loads(next_message["content"], strict=False)
                except Exception as e:
                    self.logger.debug(f"Failed to fix json. Error: {e}")
                    response_json = {"role": "assistant", "content": []}

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

    settings = {}
    client = AnthropicAPIClient(settings)
    response = client.send_message("What is the weather like today in Jakarta?")
    print(f"Response text: {response}")
    print(f"Date: {response['content']['date']}")
    print(f"Condition: {response['content']['condition']}")
