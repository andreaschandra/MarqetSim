"""Agent configuration and management."""

import copy
import json
import os
import textwrap
import uuid
from pathlib import Path
from typing import Any

import chevron
import rich
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.web import SimpleWebPageReader

from marqetsim import config
from marqetsim.environment import Environment
from marqetsim.llm.anthropic import AnthropicAPIClient
from marqetsim.llm.ollama import OllamaAPIClient
from marqetsim.llm.openai import OpenAIClient
from marqetsim.memory.rag import MarqKnowledge
from marqetsim.schema.schema import CognitiveActionModel, TinyMemory
from marqetsim.utils import common
from marqetsim.utils.common import break_text_at_length, repeat_on_error
from marqetsim.utils.logger import LogCreator


class Person:
    """Person class representing an agent in the simulation."""

    PP_TEXT_WIDTH = 100
    MAX_ACTIONS_BEFORE_DONE = 15
    communication_display: bool = True

    def __init__(self, name, logger=LogCreator("marqetsim", level="DEBUG")):
        self.name = name
        self._configuration = {"persona": {}}
        self._init_system_message = None
        self.current_messages = []
        self._mental_faculties = []
        self._displayed_communications_buffer = []
        self.logger = logger

        # default
        self.default = {}
        self.default["max_content_display_length"] = 1024
        self.default["LLM_TYPE"] = config["Simulation"].get("LLM_TYPE", "Ollama")
        self.logger.info(f"Default config: {self.default}")

        # The list of actions that this agent has performed so far, but which have not been
        # consumed by the environment yet.
        self._actions_buffer = []

        self._prompt_template_path = (
            Path(__file__).parent.parent / "prompts" / "tinyperson.mustache"
        )

        # the current environment in which the agent is acting
        self.environment = Environment()
        self._configuration["current_datetime"] = self.iso_datetime()

        # Memory
        self.episodic_memory = EpisodicMemory(name=name)
        self.semantic_memory = SemanticMemory(name="marq-knowledge")

        # LLM Provider
        self.ollama_client = OllamaAPIClient()
        self.anthropic_client = AnthropicAPIClient()

    def define(self, key, value):
        """Define Person attributes."""
        if isinstance(value, str):
            self._configuration["persona"][key] = textwrap.dedent(value)
        else:
            self._configuration["persona"][key] = value

    def set_context(self, context):
        """Set the context for the Person."""

        self._configuration["current_context"] = context

    def listen_and_act(self, message):
        """Listen to a request and act on it."""
        # Simulate listening and acting based on the request message
        self.listen(message)
        response = self.act(return_actions=True)

        return response

    # ============== listen ==============

    def listen(self, message):
        """Listen to a request."""
        stimulus = {"type": "CONVERSATION", "content": message, "source": ""}
        self._observe(stimulus=stimulus)

    def _observe(self, stimulus):
        """Observe the stimulus."""
        # Simulate observing the stimulus
        content = {"stimuli": [stimulus]}
        data = {
            "role": "user",
            "content": content,
            "type": "stimulus",
            "simulation_timestamp": self.iso_datetime(),
        }
        self.store_in_memory(data)

        if Person.communication_display:
            self._display_communication(
                role="user",
                content=content,
                kind="stimuli",
                simplified=True,
            )

        return self  # allows easier chaining of methods

    def iso_datetime(self) -> str:
        """
        Returns the current datetime of the environment, if any.

        Returns:
            datetime: The current datetime of the environment in ISO forat.
        """
        if (
            self.environment is not None
            and self.environment.current_datetime is not None
        ):
            return self.environment.current_datetime.isoformat()

    def store_in_memory(self, data):
        """Store data in memory."""
        # Simulate storing data in memory
        self.logger.debug(f"Storing data: {data}\n\n")
        self.episodic_memory.store(data)

    # ============== act ==============

    def act(self, return_actions=False):
        """
        Acts in the environment and updates its internal cognitive state.
        Either acts until the agent is done and needs additional stimuli,
        or acts a fixed number of times,
        but not both.

        Args:
            until_done (bool): Whether to keep acting until the agent is done
            and needs additional stimuli.
            n (int): The number of actions to perform. Defaults to None.
            return_actions (bool): Whether to return the actions or not. Defaults to False.
        """

        contents = []

        # Aux function to perform exactly one action.
        # Occasionally, the model will return JSON missing important keys,
        # so we just ask it to try again
        @repeat_on_error(retries=5, exceptions=[KeyError])
        def aux_act_once():

            self.logger.debug(">>>============= _produce_message() =============")
            role, content = self._produce_message()

            for subcontent in content:
                action = subcontent["action"]
                cognitive_state = subcontent["cognitive_state"]
                self.logger.debug(f"{self.name}'s action: {action}")

                self.logger.debug(">>>============= store_in_memory() =============")
                self.store_in_memory(
                    {
                        "role": role,
                        "content": subcontent,
                        "type": "action",
                        "simulation_timestamp": self.iso_datetime(),
                    }
                )

                self._actions_buffer.append(action)
                self.logger.debug(
                    ">>>============= _update_cognitive_state() ============="
                )
                self._update_cognitive_state(
                    goals=cognitive_state["goals"],
                    attention=cognitive_state["attention"],
                    emotions=cognitive_state["emotions"],
                )

                contents.append(subcontent)

                self.logger.debug(
                    ">>>============= _display_communication() ============="
                )
                if Person.communication_display:
                    self._display_communication(
                        role=role, content=subcontent, kind="action", simplified=True
                    )

                #
                # Some actions induce an immediate stimulus or other side-effects.
                # We need to process them here, by means of the mental faculties.
                #
                for faculty in self._mental_faculties:
                    faculty.process_action(self, action)

        while (len(contents) == 0) or (not contents[-1]["action"]["type"] == "DONE"):
            self.logger.debug(f"Total contents: {len(contents)}\n\n")

            # check if the agent is acting without ever stopping
            if len(contents) > Person.MAX_ACTIONS_BEFORE_DONE:
                self.logger.warning(
                    f"[{self.name}] Agent {self.name} is not stopping. This may be a bug."
                )
                break

            if len(contents) > 4:
                # just minimum number of actions to check for repetition, could be anything >= 3
                # if the last three actions were the same, then we are probably in a loop
                if (
                    contents[-1]["action"]
                    == contents[-2]["action"]
                    == contents[-3]["action"]
                ):
                    self.logger.warning(
                        f"[{self.name}] Agent {self.name} is acting in a loop. This may be a bug."
                    )
                    break

            self.logger.debug(">>============= aux_act_once() =============")
            aux_act_once()

        if return_actions:
            return contents

    def _produce_message(self):
        # logger.debug(f"Current messages: {self.current_messages}")

        # ensure we have the latest prompt (initial system message + selected messages from memory)

        self.logger.debug(">>>>============= reset_prompt() =============")
        self.reset_prompt()

        messages = [
            {"role": msg["role"], "content": json.dumps(msg["content"])}
            for msg in self.current_messages
        ]

        llm_type = self.default["LLM_TYPE"]
        self.logger.debug(f"[{self.name}] Sending messages to {llm_type}\n")
        self.logger.debug(f"[{self.name}] Messages: {messages} \n\n")

        if self.default["LLM_TYPE"] == "Ollama":
            self.logger.debug(
                ">>>>============= ollama_client.send_message() ============="
            )
            next_message = self.ollama_client.send_message(
                messages, response_format=CognitiveActionModel
            )
        elif self.default["LLM_TYPE"] == "OpenAI":
            self.logger.debug(
                ">>>>============= openai_client.send_message() ============="
            )

            openai_client = OpenAIClient()
            next_message = openai_client.send_message(
                messages, response_format=CognitiveActionModel
            )
        elif self.default["LLM_TYPE"] == "Anthropic":
            next_message = self.anthropic_client.send_message(
                messages, response_format=CognitiveActionModel
            )
        else:
            support_types = "Supported types are: Ollama, OpenAI, Anthropic."
            raise ValueError(
                f"Unknown LLM type: {self.default['LLM_TYPE']}. {support_types}"
            )

        self.logger.debug(f"[{self.name}] Received message: {next_message}\n\n")

        return next_message["role"], next_message["content"]

    def reset_prompt(self):
        """Reset prompt"""

        # render the template with the current configuration
        self.logger.debug(
            ">>>>>============= generate_agent_system_prompt() ============="
        )
        self._init_system_message = self.generate_agent_system_prompt()
        self.logger.debug("init_system_message: %s", self._init_system_message)

        # reset system message
        self.logger.debug(">>>>>=== reset_prompt: set self.current_messages ====")
        self.current_messages = [{"role": "user", "content": self._init_system_message}]

        # sets up the actual interaction messages to use for prompting
        self.logger.debug(
            ">>>>>=== reset_prompt: add self.current_messages with retrieve_recent_memories ==="
        )
        self.current_messages += self.retrieve_recent_memories()

        # add a final user message, which is neither stimuli or action,
        # to instigate the agent to act properly
        self.logger.debug(
            ">>>>>=== reset_prompt: add self.current_messages with hard coded ==="
        )
        self.current_messages.append(
            {
                "role": "user",
                "content": "**must** gen actions sequence following your interaction directives, "
                + "and comply **all** instructions and contraints related to the action you use."
                + "DO NOT repeat the exact same action more than once in a row!"
                + "These actions **MUST** be rendered following the JSON specification perfectly,"
                + "including all required keys (even if their value is empty), **ALWAYS**.",
            }
        )

    def generate_agent_system_prompt(self):
        """Generate agent system prompt."""

        with open(self._prompt_template_path, encoding="utf-8", mode="r") as f:
            agent_prompt_template = f.read()

        # let's operate on top of a copy of the configuration,
        # because we'll need to add more variables, etc.
        template_variables = self._configuration.copy()

        # RAI prompt components, if requested
        template_variables = common.add_rai_template_variables_if_enabled(
            template_variables
        )

        return chevron.render(agent_prompt_template, template_variables)

    def retrieve_recent_memories(self, max_content_length: int = None) -> list:
        """retrieve recent memories"""
        episodes = self.episodic_memory.retrieve_recent()

        if max_content_length is not None:
            episodes = common.truncate_actions_or_stimuli(episodes, max_content_length)

        return episodes

    def _update_cognitive_state(
        self, goals=None, context=None, attention=None, emotions=None
    ):
        """
        Update the TinyPerson's cognitive state.
        """

        # Update current datetime. The passage of time is controlled by the environment, if any.
        if (
            self.environment is not None
            and self.environment.current_datetime is not None
        ):
            self._configuration["current_datetime"] = common.pretty_datetime(
                self.environment.current_datetime
            )

        # update current goals
        if goals is not None:
            self._configuration["current_goals"] = goals

        # update current context
        if context is not None:
            self._configuration["current_context"] = context

        # update current attention
        if attention is not None:
            self._configuration["current_attention"] = attention

        # update current emotions
        if emotions is not None:
            self._configuration["current_emotions"] = emotions

        # update relevant memories for the current situation

        current_memory_context = self.retrieve_relevant_memories_for_current_context()
        self._configuration["current_memory_context"] = current_memory_context

        self.reset_prompt()

    def retrieve_relevant_memories_for_current_context(self, top_k=7) -> list:
        """Retrieve relevant memories."""

        # current context is composed of the recent memories, plus context, goals, attention, and emotions
        context = self._configuration["current_context"]
        goals = self._configuration["current_goals"]
        attention = self._configuration["current_attention"]
        emotions = self._configuration["current_emotions"]
        recent_memories = "\n".join(
            [
                f"  - {m['content']}"
                for m in self.retrieve_memories(
                    first_n=0, last_n=10, max_content_length=100
                )
            ]
        )

        # put everything together in a nice markdown string to fetch relevant memories
        target = f"""
        Current Context: {context}
        Current Goals: {goals}
        Current Attention: {attention}
        Current Emotions: {emotions}
        Recent Memories:
        {recent_memories}
        """

        self.logger.debug(
            f"Retrieving relevant memories for contextual target: {target}"
        )

        return self.retrieve_relevant_memories(target, top_k=top_k)

    def retrieve_memories(
        self,
        first_n: int,
        last_n: int,
        include_omission_info: bool = True,
        max_content_length: int = None,
    ) -> list:
        """Retrieve memories from episodic."""

        episodes = self.episodic_memory.retrieve(
            first_n=first_n, last_n=last_n, include_omission_info=include_omission_info
        )

        if max_content_length is not None:
            episodes = common.truncate_actions_or_stimuli(episodes, max_content_length)

        return episodes

    def retrieve_relevant_memories(self, relevance_target: str, top_k=20) -> list:
        """Retrieve relevant memories."""

        relevant = self.semantic_memory.retrieve_relevant(relevance_target, top_k=top_k)

        return relevant

    # ============== Display ==============
    def _display_communication(self, role, content, kind, simplified=True):
        """
        Displays the current communication and stores it in a buffer for later use.
        """
        if kind == "stimuli":
            rendering = self._pretty_stimuli(
                role=role, content=content, simplified=simplified
            )
            source = content["stimuli"][0]["source"]
            target = self.name

        elif kind == "action":
            rendering = self._pretty_action(
                role=role, content=content, simplified=simplified
            )
            source = self.name
            target = content["action"]["target"]

        else:
            raise ValueError(f"Unknown communication kind: {kind}")

        self._push_and_display_latest_communication(
            {
                "kind": kind,
                "rendering": rendering,
                "content": content,
                "source": source,
                "target": target,
            }
        )

    def _pretty_stimuli(
        self,
        role,
        content,
        simplified=True,
    ) -> list:
        """
        Pretty prints stimuli.
        """

        lines = []
        msg_simplified_actor = "USER"
        for stimus in content["stimuli"]:
            if simplified:
                if stimus["source"] != "":
                    msg_simplified_actor = stimus["source"]

                else:
                    msg_simplified_actor = "USER"

                msg_simplified_type = stimus["type"]
                msg_simplified_content = break_text_at_length(
                    stimus["content"],
                    max_length=self.default.get("max_content_display_length", 1024),
                )

                indent = " " * len(msg_simplified_actor) + "      > "
                msg_simplified_content = textwrap.fill(
                    msg_simplified_content,
                    width=Person.PP_TEXT_WIDTH,
                    initial_indent=indent,
                    subsequent_indent=indent,
                )

                #
                # Using rich for formatting. Let's make things as readable as possible!
                #

                rich_style = common.RichTextStyle.get_style_for(
                    "stimulus", msg_simplified_type
                )
                lines.append(
                    f"[{rich_style}][underline]{msg_simplified_actor}[/] --> [{rich_style}][underline]{self.name}[/]: [{msg_simplified_type}] \n{msg_simplified_content}[/]"
                )
            else:
                lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _pretty_action(self, role, content, simplified=True) -> str:
        """
        Pretty prints an action.
        """
        if simplified:
            msg_simplified_actor = self.name
            msg_simplified_type = content["action"]["type"]
            msg_simplified_content = break_text_at_length(
                content["action"].get("content", ""),
                max_length=self.default.get("max_content_display_length", 1024),
            )

            indent = " " * len(msg_simplified_actor) + "      > "
            msg_simplified_content = textwrap.fill(
                msg_simplified_content,
                width=Person.PP_TEXT_WIDTH,
                initial_indent=indent,
                subsequent_indent=indent,
            )

            #
            # Using rich for formatting. Let's make things as readable as possible!
            #
            rich_style = common.RichTextStyle.get_style_for(
                "action", msg_simplified_type
            )
            return f"[{rich_style}][underline]{msg_simplified_actor}[/] acts: [{msg_simplified_type}] \n{msg_simplified_content}[/]"

        else:
            return f"{role}: {content}"

    def _push_and_display_latest_communication(self, communication):
        """
        Pushes the latest communications to the agent's buffer.
        """
        self._displayed_communications_buffer.append(communication)
        rich.print(communication["rendering"])


class EpisodicMemory(TinyMemory):
    """
    Provides episodic memory capabilities to an agent. Cognitively, episodic memory is
    the ability to remember specific events,
    or episodes, in the past. This class provides a simple implementation of episodic memory,
    where the agent can store and retrieve
    messages from memory.

    Subclasses of this class can be used to provide different memory implementations.
    """

    MEMORY_BLOCK_OMISSION_INFO = {
        "role": "assistant",
        "content": "Info: there were other messages here, but they were omitted for brevity.",
        "simulation_timestamp": None,
    }

    def __init__(
        self,
        name: str = "agent",
        fixed_prefix_length: int = 100,
        lookback_length: int = 100,
    ) -> None:
        """
        Initializes the memory.

        Args:
            fixed_prefix_length (int): The fixed prefix length. Defaults to 20.
            lookback_length (int): The lookback length. Defaults to 20.
        """
        super().__init__(name=name)
        self.fixed_prefix_length = fixed_prefix_length
        self.lookback_length = lookback_length

        self.memory = []

    def _store(self, value: Any) -> None:
        """
        Stores a value in memory.
        """
        self.memory.append(value)

    def count(self) -> int:
        """
        Returns the number of values in memory.
        """
        return len(self.memory)

    def retrieve(
        self, first_n: int, last_n: int, include_omission_info: bool = True
    ) -> list:
        """
        Retrieves the first n and/or last n values from memory. If n is None, all values are retrieved.

        Args:
            first_n (int): The number of first values to retrieve.
            last_n (int): The number of last values to retrieve.
            include_omission_info (bool): Whether to include an information message when some values are omitted.

        Returns:
            list: The retrieved values.

        """

        omisssion_info = (
            [EpisodicMemory.MEMORY_BLOCK_OMISSION_INFO] if include_omission_info else []
        )

        # use the other methods in the class to implement
        if first_n is not None and last_n is not None:
            return (
                self.retrieve_first(first_n)
                + omisssion_info
                + self.retrieve_last(last_n)
            )
        elif first_n is not None:
            return self.retrieve_first(first_n)
        elif last_n is not None:
            return self.retrieve_last(last_n)
        else:
            return self.retrieve_all()

    def retrieve_recent(self, include_omission_info: bool = True) -> list:
        """
        Retrieves the n most recent values from memory.
        """
        omisssion_info = (
            [EpisodicMemory.MEMORY_BLOCK_OMISSION_INFO] if include_omission_info else []
        )

        # compute fixed prefix
        fixed_prefix = self.memory[: self.fixed_prefix_length] + omisssion_info

        # how many lookback values remain?
        remaining_lookback = min(
            len(self.memory) - len(fixed_prefix), self.lookback_length
        )

        # compute the remaining lookback values and return the concatenation
        if remaining_lookback <= 0:
            return fixed_prefix
        else:
            return fixed_prefix + self.memory[-remaining_lookback:]

    def retrieve_all(self) -> list:
        """
        Retrieves all values from memory.
        """
        return copy.copy(self.memory)

    def retrieve_relevant(self, relevance_target: str, top_k: int = 20) -> list:
        """
        Retrieves top-k values from memory that are most relevant to a given target.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def retrieve_first(self, n: int, include_omission_info: bool = True) -> list:
        """
        Retrieves the first n values from memory.
        """
        omisssion_info = (
            [EpisodicMemory.MEMORY_BLOCK_OMISSION_INFO] if include_omission_info else []
        )

        return self.memory[:n] + omisssion_info

    def retrieve_last(self, n: int, include_omission_info: bool = True) -> list:
        """
        Retrieves the last n values from memory.
        """
        omisssion_info = (
            [EpisodicMemory.MEMORY_BLOCK_OMISSION_INFO] if include_omission_info else []
        )

        return omisssion_info + self.memory[-n:]


class SemanticMemory(TinyMemory):
    """
    Semantic memory is the memory of meanings, understandings, and other concept-based knowledge
    unrelated to specific experiences.
    It is not ordered temporally, and it is not about remembering specific events or episodes.
    This class provides a simple implementation
    of semantic memory, where the agent can store and retrieve semantic information.
    """

    suppress_attributes_from_serialization = ["index"]

    def __init__(
        self,
        documents_paths: list = None,
        web_urls: list = None,
        based_knowledge=None,
        name=None,
        persistent_path=None,
    ) -> None:
        super().__init__(name=name)
        self.based_knowledge = based_knowledge or MarqKnowledge(
            collection_name=name, persist_directory=persistent_path
        )

        self.documents = []
        self.documents_paths = []
        self.documents_web_urls = []
        self.filename_to_document = {}

        self.add_documents_paths(documents_paths)
        if web_urls:
            self.add_web_urls(web_urls)

    def _preprocess_value_for_storage(self, value: dict) -> Any:
        engram = None
        ts = value["simulation_timestamp"]

        if value["type"] == "action":
            engram = (
                "# Fact\n"
                + f"I have performed the following action at date and time {ts}:\n\n"
                + f" {value['content']}"
            )

        elif value["type"] == "stimulus":
            engram = (
                "# Stimulus\n"
                + f"I have received the following stimulus at date and time {ts}:\n\n"
                + f" {value['content']}"
            )

        # else: # Anything else here?

        return engram

    def _store(self, value: Any) -> None:
        # engram_doc = Document(text=str(value))
        engram_text = str(value)
        doc_id = str(uuid.uuid4())
        self.based_knowledge.add_document(engram_text, doc_id)

    def retrieve_relevant(self, relevance_target: str, top_k=20) -> list:
        """
        Retrieves all values from memory that are relevant to a given target.
        """
        results = self.based_knowledge.retrieve(relevance_target, top_k)
        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            content = f"SOURCE: {meta.get('file_name', '(unknown)')}\n"
            content += f"DISTANCE: {dist:.4f}\n"
            content += f"RELEVANT CONTENT:\n{doc}"
            retrieved.append(content)
        return retrieved

    def retrieve_document_content_by_name(self, document_name: str) -> str:
        """
        Retrieves a document by its name.
        """
        if self.filename_to_document is not None:
            doc = self.filename_to_document[document_name]
            if doc is not None:
                content = "SOURCE: " + document_name
                content += "\n" + "CONTENT: " + doc.text[:10000]
                return content
            else:
                return None
        else:
            return None

    def list_documents_names(self) -> list:
        """
        Lists the names of the documents in memory.
        """
        if self.filename_to_document is not None:
            return list(self.filename_to_document.keys())
        else:
            return []

    def add_documents_paths(self, documents_paths: list) -> None:
        """
        Adds a path to a folder with documents used for semantic memory.
        """

        if documents_paths is not None:
            for documents_path in documents_paths:
                try:
                    self.add_documents_path(documents_path)
                except (FileNotFoundError, ValueError) as e:
                    rich.print(f"Error: {e}")
                    rich.print(f"Current working directory: {os.getcwd()}")
                    rich.print(f"Provided path: {documents_path}")
                    rich.print("Please check if the path exists and is accessible.")

    def add_documents_path(self, documents_path: str) -> None:
        """
        Adds a path to a folder with documents used for semantic memory.
        """

        documents = SimpleDirectoryReader(input_dir=documents_path).load_data()
        for doc in documents:
            sanitized_text = common.sanitize_raw_string(doc.text)
            doc_id = str(uuid.uuid4())
            file_name = doc.metadata.get("file_name", "unknown")
            self.filename_to_document[file_name] = doc
            self.based_knowledge.add_document(
                sanitized_text, doc_id, {"file_name": file_name}
            )

    def add_web_urls(self, web_urls: list) -> None:
        """
        Adds the data retrieved from the specified URLs to documents used for semantic memory.
        """

        filtered_urls = [url for url in web_urls if url not in self.documents_web_urls]
        self.documents_web_urls += filtered_urls

        if filtered_urls:
            self.add_web_urls(filtered_urls)

    def add_web_url(self, web_urls: str) -> None:
        """
        Adds the data retrieved from the specified URL to documents used for semantic memory.
        """
        # we do it like this because the add_web_urls could run scrapes in parallel, so it is better
        # to implement this one in terms of the other

        # self.add_web_urls([web_url])
        documents = SimpleWebPageReader(html_to_text=True).load_data(web_urls)
        for doc in documents:
            doc.text = common.sanitize_raw_string(doc.text)
            doc_id = str(uuid.uuid4())
            self.based_knowledge.add_document(doc.text, doc_id, {"source": "web"})

    def _add_documents(self, new_documents, doc_to_name_func=None) -> list:
        """
        Adds multiple documents by calling _add_document on each.
        """
        for document in new_documents:
            self._add_document(document, doc_to_name_func)

    def _add_document(self, document, doc_to_name_func=None) -> None:
        """
        Adds a single document to the semantic memory.
        """
        # Sanitize text
        document.text = common.sanitize_raw_string(document.text)

        # Determine document name if function provided
        name = None
        if doc_to_name_func is not None:
            name = doc_to_name_func(document)
            self.filename_to_document[name] = document

        # Generate unique doc_id
        doc_id = str(uuid.uuid4())

        # Add to vector DB
        self.based_knowledge.add_document(
            document.text, doc_id, metadata={"file_name": name}
        )

        # Also keep in documents list
        self.documents.append(document)

    ###########################################################
    # IO
    ###########################################################

    def _post_deserialization_init(self):

        # Reset or recreate MarqKnowledge instance if needed
        self.based_knowledge = MarqKnowledge()

        # Reload documents and web URLs into MarqKnowledge
        self.add_documents_paths(self.documents_paths)
        self.add_web_urls(self.documents_web_urls)

    def actions_constraints_prompt(self):
        return None

    def actions_definitions_prompt(self):
        return None

    def process_action(self, agent, action):
        return None
