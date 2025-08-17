"""OpenAI API client for interacting with OpenAI and Azure OpenAI services."""

import logging
import os
import pickle
import time

import openai
import tiktoken
from openai import AzureOpenAI, OpenAI

from marqetsim.utils import common

logger = logging.getLogger("marqetsim")

config = common.read_config_file()

###########################################################################
# Default parameter values
###########################################################################
default = {}
default["model"] = config["OpenAI"].get("MODEL", "gpt-4o")
default["max_tokens"] = int(config["OpenAI"].get("MAX_TOKENS", "1024"))
default["temperature"] = float(config["OpenAI"].get("TEMPERATURE", "1.0"))
default["top_p"] = int(config["OpenAI"].get("TOP_P", "0"))
default["frequency_penalty"] = float(config["OpenAI"].get("FREQ_PENALTY", "0.0"))
default["presence_penalty"] = float(config["OpenAI"].get("PRESENCE_PENALTY", "0.0"))
default["timeout"] = float(config["OpenAI"].get("TIMEOUT", "30.0"))
default["max_attempts"] = float(config["OpenAI"].get("MAX_ATTEMPTS", "0.0"))
default["waiting_time"] = float(config["OpenAI"].get("WAITING_TIME", "1"))
default["exponential_backoff_factor"] = float(
    config["OpenAI"].get("EXPONENTIAL_BACKOFF_FACTOR", "5")
)

default["embedding_model"] = config["OpenAI"].get(
    "EMBEDDING_MODEL", "text-embedding-3-small"
)

default["cache_api_calls"] = config["OpenAI"].getboolean("CACHE_API_CALLS", False)
default["cache_file_name"] = config["OpenAI"].get(
    "CACHE_FILE_NAME", "openai_api_cache.pickle"
)


###########################################################################
# Client class
###########################################################################


class OpenAIClient:
    """
    A utility class for interacting with the OpenAI API.
    """

    def __init__(
        self,
        cache_api_calls=default["cache_api_calls"],
        cache_file_name=default["cache_file_name"],
    ) -> None:
        logger.debug("Initializing OpenAIClient")

        # should we cache api calls and reuse them?
        self.set_api_cache(cache_api_calls, cache_file_name)
        self.client = None

    def set_api_cache(
        self, cache_api_calls, cache_file_name=default["cache_file_name"]
    ):
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
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def send_message(
        self,
        current_messages,
        default=default,
        stop=None,
        n=1,
        response_format=None,
    ):
        """
        Sends a message to the OpenAI API and returns the response.

        Args:
        current_messages (list): A list of dictionaries representing the conversation history.
        default (dict): A dictionary containing the default parameters for the API call.
        stop (str): A string that, if encountered in the generated response, will cause the generation to stop.
        n (int): The number of completions to generate.
        response_format (str): The format of the response. If None, the response is returned as a dictionary.

        Returns:
        A dictionary representing the generated response.
        """

        model = default.get("model")
        waiting_time = default.get("waiting_time")
        exponential_backoff_factor = default.get("exponential_backoff_factor")

        def aux_exponential_backoff():
            nonlocal waiting_time

            # in case waiting time was initially set to 0
            if waiting_time <= 0:
                waiting_time = 2

            logger.info(
                f"Request failed. Waiting {waiting_time} seconds between requests..."
            )
            time.sleep(waiting_time)

            # exponential backoff
            waiting_time = waiting_time * exponential_backoff_factor

        # setup the OpenAI configurations for this client.
        self._setup_from_config()

        # We need to adapt the parameters to the API type, so we create a dictionary with them first
        chat_api_params = {
            "model": default.get("model"),
            "messages": current_messages,
            "temperature": default.get("temperature"),
            "max_tokens": default.get("max_tokens"),
            "top_p": default.get("top_p"),
            "frequency_penalty": default.get("frequency_penalty"),
            "presence_penalty": default.get("presence_penalty"),
            "stop": stop or [],
            "timeout": default.get("timeout"),
            "stream": False,
            "n": n,
        }

        if response_format is not None:
            chat_api_params["response_format"] = response_format

        i = 0
        while i < default.get("max_attempts"):
            try:
                i += 1

                try:
                    logger.debug(
                        f"Sending messages to OpenAI API. Token count={self._count_tokens(current_messages, model)}."
                    )
                except NotImplementedError:
                    logger.debug(f"Token count not implemented for model {model}.")

                start_time = time.monotonic()
                logger.debug(
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
                        logger.info(
                            f"Waiting {waiting_time} seconds before next API request (to avoid throttling)..."
                        )
                        time.sleep(waiting_time)

                    logger.debug(">>>>>============= _raw_model_call() =============")
                    response = self._raw_model_call(model, chat_api_params)
                    if self.cache_api_calls:
                        self.api_cache[cache_key] = response
                        self._save_cache()

                logger.debug(f"Got response from API: {response}")
                end_time = time.monotonic()
                logger.debug(
                    f"Got response in {end_time - start_time:.2f} seconds after {i} attempts."
                )

                return common.sanitize_dict(
                    self._raw_model_response_extractor(response)
                )

            except InvalidRequestError as e:
                logger.error(f"[{i}] Invalid request error, won't retry: {e}")

                # there's no point in retrying if the request is invalid
                # so we return None right away
                return None

            except openai.BadRequestError as e:
                logger.error(f"[{i}] Invalid request error, won't retry: {e}")

                # there's no point in retrying if the request is invalid
                # so we return None right away
                return None

            except openai.RateLimitError:
                logger.warning(
                    f"[{i}] Rate limit error, waiting a bit and trying again."
                )
                aux_exponential_backoff()

            except NonTerminalError as e:
                logger.error(f"[{i}] Non-terminal error: {e}")
                aux_exponential_backoff()

            except Exception as e:
                logger.error(f"[{i}] Error: {e}")

        logger.error(
            f"Failed to get response after {default.get('max_attempts')} attempts."
        )
        return None

    def _raw_model_call(self, chat_api_params):
        """
        Calls the OpenAI API with the given parameters. Subclasses should
        override this method to implement their own API calls.
        """

        if "response_format" in chat_api_params:
            # to enforce the response format, we need to use a different method
            logger.debug(
                ">>>>>>========== self.client.beta.chat.completions.parse =========="
            )
            logger.debug(
                f">>>>>>========== chat_api_params.response_format: {chat_api_params['response_format']} =========="
            )
            del chat_api_params["stream"]

            return self.client.beta.chat.completions.parse(**chat_api_params)

        logger.debug(">>>>>>========== self.client.chat.completions.create ==========")
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
                logger.debug(
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
                logger.debug(
                    "Token count: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
                )
                return self._count_tokens(messages, model="gpt-3.5-turbo-0613")
            elif ("gpt-4" in model) or ("ppo" in model):
                logger.debug(
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
            logger.error(f"Error counting tokens: {e}")
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

    def get_embedding(self, text, model=default["embedding_model"]):
        """
        Gets the embedding of the given text using the specified model.

        Args:
        text (str): The text to embed.
        model (str): The name of the model to use for embedding the text.

        Returns:
        The embedding of the text.
        """
        response = self._raw_embedding_model_call(text, model)
        return self._raw_embedding_model_response_extractor(response)

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


class AzureClient(OpenAIClient):
    """A utility class for interacting with the Azure OpenAI Service API.
    This class extends the OpenAIClient to support Azure-specific configurations.
    """

    def __init__(
        self,
        cache_api_calls=default["cache_api_calls"],
        cache_file_name=default["cache_file_name"],
    ) -> None:
        logger.debug("Initializing AzureClient")

        super().__init__(cache_api_calls, cache_file_name)

    def _setup_from_config(self):
        """
        Sets up the Azure OpenAI Service API configurations for this client,
        including the API endpoint and key.
        """
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=config["OpenAI"]["AZURE_API_VERSION"],
            api_key=os.getenv("AZURE_OPENAI_KEY"),
        )


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


###########################################################################
# Clients registry
#
# We can have potentially different clients, so we need a place to
# register them and retrieve them when needed.
#
# We support both OpenAI and Azure OpenAI Service API by default.
# Thus, we need to set the API parameters based on the choice of the user.
# This is done within specialized classes.
#
# It is also possible to register custom clients, to access internal or
# otherwise non-conventional API endpoints.
###########################################################################
_api_type_to_client = {}
_API_TYPE_OVERRIDE = None


def register_client(api_type, client):
    """
    Registers a client for the given API type.

    Args:
    api_type (str): The API type for which we want to register the client.
    client: The client to register.
    """
    _api_type_to_client[api_type] = client


def _get_client_for_api_type(api_type):
    """
    Returns the client for the given API type.

    Args:
    api_type (str): The API type for which we want to get the client.
    """
    try:
        return _api_type_to_client[api_type]
    except KeyError as e:
        raise ValueError(
            f"API type {api_type} is not supported. Please check the 'config.ini' file."
        ) from e


def client():
    """
    Returns the client for the configured API type.
    """
    api_type = (
        config["OpenAI"]["API_TYPE"]
        if _API_TYPE_OVERRIDE is None
        else _API_TYPE_OVERRIDE
    )

    logger.debug(f"Using  API type {api_type}.")
    return _get_client_for_api_type(api_type)


def force_api_type(api_type):
    """
    Forces the use of the given API type, thus overriding any other configuration.

    Args:
    api_type (str): The API type to use.
    """
    global _API_TYPE_OVERRIDE
    _API_TYPE_OVERRIDE = api_type


def force_api_cache(cache_api_calls, cache_file_name=default["cache_file_name"]):
    """
    Forces the use of the given API cache configuration, thus overriding any other configuration.

    Args:
    cache_api_calls (bool): Whether to cache API calls.
    cache_file_name (str): The name of the file to use for caching API calls.
    """
    # set the cache parameters on all clients
    for client in _api_type_to_client.values():
        client.set_api_cache(cache_api_calls, cache_file_name)


# default client
register_client("openai", OpenAIClient())
register_client("azure", AzureClient())
