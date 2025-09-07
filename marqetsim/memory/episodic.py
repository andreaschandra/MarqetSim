"""Episodic memory module."""

import copy
from typing import Any
from marqetsim.memory.base import TinyMemory


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

    def __init__(self):

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

    def _retrieve_first(self, n: int, include_omission_info: bool = True) -> list:
        """
        Retrieves the first n values from memory.
        """
        omisssion_info = (
            [EpisodicMemory.MEMORY_BLOCK_OMISSION_INFO] if include_omission_info else []
        )

        return self.memory[:n] + omisssion_info

    def _retrieve_last(self, n: int, include_omission_info: bool = True) -> list:
        """
        Retrieves the last n values from memory.
        """
        omisssion_info = (
            [EpisodicMemory.MEMORY_BLOCK_OMISSION_INFO] if include_omission_info else []
        )

        return omisssion_info + self.memory[-n:]

    def retrieve(
        self,
        first_n: int = None,
        last_n: int = None,
        include_omission_info: bool = True,
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
                self._retrieve_first(first_n)
                + omisssion_info
                + self._retrieve_last(last_n)
            )
        elif first_n is not None:
            return self._retrieve_first(first_n)
        elif last_n is not None:
            return self._retrieve_last(last_n)
        else:
            return self.retrieve_all()

    def retrieve_recent(self, n=5) -> list:
        """
        Retrieves the n most recent values from memory.
        """

        return self._retrieve_last(n=n)

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
