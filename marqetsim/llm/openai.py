"""OpenAI API client for interacting with OpenAI and Azure OpenAI services."""

import os
import pickle
import time

import openai
import tiktoken
from openai import OpenAI

from marqetsim import config
from marqetsim.llm.base import LLMBase
from marqetsim.utils import LogCreator, common


class OpenAIClient(LLMBase):
    """
    A utility class for interacting with the OpenAI API.
    """

    def __init__(
        self,
        settings,
        logger=LogCreator(),
    ):

        self.logger = logger
        self.logger.debug("Initializing OpenAIClient")

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.settings = settings
        self.default = None
        self.set_defaults()
        self.set_api_cache(
            self.default["cache_api_calls"], self.default["cache_file_name"]
        )

    def set_defaults(self):
        """Set default config for model."""

        default = {}
        default["model"] = config["OpenAI"].get("MODEL", "gpt-4o")
        default["max_tokens"] = int(config["OpenAI"].get("MAX_TOKENS", "1024"))
        default["temperature"] = float(config["OpenAI"].get("TEMPERATURE", "1.0"))
        default["top_p"] = int(config["OpenAI"].get("TOP_P", "0"))
        default["frequency_penalty"] = float(
            config["OpenAI"].get("FREQ_PENALTY", "0.0")
        )
        default["presence_penalty"] = float(
            config["OpenAI"].get("PRESENCE_PENALTY", "0.0")
        )
        default["timeout"] = float(config["OpenAI"].get("TIMEOUT", "30.0"))
        default["max_attempts"] = float(config["OpenAI"].get("MAX_ATTEMPTS", "0.0"))
        default["waiting_time"] = float(config["OpenAI"].get("WAITING_TIME", "1"))
        default["exponential_backoff_factor"] = float(
            config["OpenAI"].get("EXPONENTIAL_BACKOFF_FACTOR", "5")
        )
        default["embedding_model"] = config["OpenAI"].get(
            "EMBEDDING_MODEL", "text-embedding-3-small"
        )
        default["cache_api_calls"] = config["OpenAI"].getboolean(
            "CACHE_API_CALLS", False
        )
        default["cache_file_name"] = config["OpenAI"].get(
            "CACHE_FILE_NAME", "openai_api_cache.pickle"
        )
        self.default = default

    def set_api_cache(self, cache_api_calls, cache_file_name):
        """
        Enables or disables the caching of API calls.

        Args:
        cache_file_name (str): The name of the file to use for caching API calls.
        """

        self.cache_api_calls = cache_api_calls
        self.cache_file_name = cache_file_name
        if self.cache_api_calls:
            # load the cache, if any
            self.api_cache = self._load_cache()

    def _setup_from_config(self):
        """
        Sets up the OpenAI API configurations for this client.
        """

    def send_message(
        self,
        messages,
        system_message=None,
        stop=None,
        n=1,
        response_format=None,
    ):
        """
        Sends a message to the OpenAI API and returns the response.

        Args:
        messages (list): A list of dictionaries representing the conversation history.
        default (dict): A dictionary containing the default parameters for the API call.
        stop (str): A string that, if encountered in the generated response, will cause the generation to stop.
        n (int): The number of completions to generate.
        response_format (str): The format of the response. If None, the response is returned as a dictionary.

        Returns:
        A dictionary representing the generated response.
        """

        model = self.default.get("model")
        waiting_time = self.default.get("waiting_time")
        exponential_backoff_factor = self.default.get("exponential_backoff_factor")

        if system_message:
            messages = [{"role": "system", "content": system_message}] + messages

        def aux_exponential_backoff():
            nonlocal waiting_time

            # in case waiting time was initially set to 0
            if waiting_time <= 0:
                waiting_time = 2

            self.logger.info(
                f"Request failed. Waiting {waiting_time} seconds between requests..."
            )
            time.sleep(waiting_time)

            # exponential backoff
            waiting_time = waiting_time * exponential_backoff_factor

        # We need to adapt the parameters to the API type, so we create a dictionary with them first
        chat_api_params = {
            "model": self.default.get("model"),
            "messages": messages,
            "temperature": self.default.get("temperature"),
            "max_tokens": self.default.get("max_tokens"),
            "top_p": self.default.get("top_p"),
            "frequency_penalty": self.default.get("frequency_penalty"),
            "presence_penalty": self.default.get("presence_penalty"),
            "stop": stop or [],
            "timeout": self.default.get("timeout"),
            "stream": False,
            "n": n,
        }

        if response_format is not None:
            chat_api_params["response_format"] = response_format

        i = 0
        while i < self.default.get("max_attempts"):
            try:
                i += 1

                try:
                    self.logger.debug(
                        f"Sending messages to OpenAI API. Token count={self._count_tokens(messages, model)}."
                    )
                except NotImplementedError:
                    self.logger.debug(f"Token count not implemented for model {model}.")

                start_time = time.monotonic()
                self.logger.debug(
                    f"Calling model with client class {self.__class__.__name__}."
                )

                ###############################################################
                # call the model, either from the cache or from the API
                ###############################################################
                cache_key = str((model, chat_api_params))  # need string to be hashable
                if self.cache_api_calls and (cache_key in self.api_cache):
                    response = self.api_cache[cache_key]
                else:
                    if waiting_time > 0:
                        self.logger.info(
                            f"Waiting {waiting_time} seconds before next API request (to avoid throttling)..."
                        )
                        time.sleep(waiting_time)

                    self.logger.debug(
                        ">>>>>============= _raw_model_call() ============="
                    )
                    response = self._raw_model_call(model, chat_api_params)
                    if self.cache_api_calls:
                        self.api_cache[cache_key] = response
                        self._save_cache()

                self.logger.debug(f"Got response from API: {response}")
                end_time = time.monotonic()
                self.logger.debug(
                    f"Got response in {end_time - start_time:.2f} seconds after {i} attempts."
                )

                return common.sanitize_dict(
                    self._raw_model_response_extractor(response)
                )

            except InvalidRequestError as e:
                self.logger.error(f"[{i}] Invalid request error, won't retry: {e}")

                # there's no point in retrying if the request is invalid
                # so we return None right away
                return None

            except openai.BadRequestError as e:
                self.logger.error(f"[{i}] Invalid request error, won't retry: {e}")

                # there's no point in retrying if the request is invalid
                # so we return None right away
                return None

            except openai.RateLimitError:
                self.logger.warning(
                    f"[{i}] Rate limit error, waiting a bit and trying again."
                )
                aux_exponential_backoff()

            except NonTerminalError as e:
                self.logger.error(f"[{i}] Non-terminal error: {e}")
                aux_exponential_backoff()

            except Exception as e:
                self.logger.error(f"[{i}] Error: {e}")

        self.logger.error(
            f"Failed to get response after {self.default.get('max_attempts')} attempts."
        )
        return None

    def _raw_model_call(self, chat_api_params):
        """
        Calls the OpenAI API with the given parameters. Subclasses should
        override this method to implement their own API calls.
        """

        if "response_format" in chat_api_params:
            # to enforce the response format, we need to use a different method
            self.logger.debug(
                ">>>>>>========== self.client.beta.chat.completions.parse =========="
            )
            self.logger.debug(
                f">>>>>>========== chat_api_params.response_format: {chat_api_params['response_format']} =========="
            )
            del chat_api_params["stream"]

            return self.client.beta.chat.completions.parse(**chat_api_params)

        self.logger.debug(
            ">>>>>>========== self.client.chat.completions.create =========="
        )
        return self.client.chat.completions.create(**chat_api_params)

    def _raw_model_response_extractor(self, response):
        """
        Extracts the response from the API response. Subclasses should
        override this method to implement their own response extraction.
        """
        return response.choices[0].message.to_dict()

    def _count_tokens(self, messages: list, model: str):
        """
        Count the number of OpenAI tokens in a list of messages using tiktoken.

        Adapted from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

        Args:
        messages (list): A list of dictionaries representing the conversation history.
        model (str): The name of the model to use for encoding the string.
        """
        try:
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                self.logger.debug(
                    "Token count: model not found. Using cl100k_base encoding."
                )
                encoding = tiktoken.get_encoding("cl100k_base")
            if model in {
                "gpt-3.5-turbo-0613",
                "gpt-3.5-turbo-16k-0613",
                "gpt-4-0314",
                "gpt-4-32k-0314",
                "gpt-4-0613",
                "gpt-4-32k-0613",
            }:
                tokens_per_message = 3
                tokens_per_name = 1
            elif model == "gpt-3.5-turbo-0301":
                tokens_per_message = (
                    4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
                )
                tokens_per_name = -1  # if there's a name, the role is omitted
            elif "gpt-3.5-turbo" in model:
                self.logger.debug(
                    "Token count: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
                )
                return self._count_tokens(messages, model="gpt-3.5-turbo-0613")
            elif ("gpt-4" in model) or ("ppo" in model):
                self.logger.debug(
                    "Token count: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
                )
                return self._count_tokens(messages, model="gpt-4-0613")
            else:
                raise NotImplementedError(
                    f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
                )
            num_tokens = 0
            for message in messages:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
            num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
            return num_tokens

        except Exception as e:
            self.logger.error(f"Error counting tokens: {e}")
            return None

    def _save_cache(self):
        """
        Saves the API cache to disk. We use pickle to do that because some obj
        are not JSON serializable.
        """
        # use pickle to save the cache
        with open(self.cache_file_name, "wb") as f:
            pickle.dump(self.api_cache, f)

    def _load_cache(self):
        """
        Loads the API cache from disk.
        """
        # unpickle
        if os.path.exists(self.cache_file_name):
            with open(self.cache_file_name, "rb") as f:
                return pickle.load(f)
        return {}

    def _raw_embedding_model_call(self, text, model):
        """
        Calls the OpenAI API to get the embedding of the given text. Subclasses should
        override this method to implement their own API calls.
        """
        return self.client.embeddings.create(input=[text], model=model)

    def _raw_embedding_model_response_extractor(self, response):
        """
        Extracts the embedding from the API response. Subclasses should
        override this method to implement their own response extraction.
        """
        return response.data[0].embedding


###########################################################################
# Exceptions
###########################################################################
class InvalidRequestError(Exception):
    """
    Exception raised when the request to the OpenAI API is invalid.
    """


class NonTerminalError(Exception):
    """
    Exception raised when an unspecified error occurs but we know we can retry.
    """
