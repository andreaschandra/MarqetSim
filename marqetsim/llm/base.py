"""Base Class for LLM Provider."""

from abc import ABC, abstractmethod


class LLMBase(ABC):
    """Base class of LLM provider."""

    @abstractmethod
    def send_message(self, messages, system_message):
        """Send message to llm model."""
