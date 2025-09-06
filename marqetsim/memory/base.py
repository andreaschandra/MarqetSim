"""Base class for TinyMentalFaculty and TinyMemory."""

from typing import Any


class TinyMentalFaculty:
    """
    Represents a mental faculty of an agent. Mental faculties are the cognitive abilities that an agent has.
    """

    def __init__(self, name: str, requires_faculties: list = None) -> None:
        """
        Initializes the mental faculty.

        Args:
            name (str): The name of the mental faculty.
            requires_faculties (list): A list of mental faculties that this faculty requires to function properly.
        """
        self.name = name

        if requires_faculties is None:
            self.requires_faculties = []
        else:
            self.requires_faculties = requires_faculties

    def __str__(self) -> str:
        return f"Mental Faculty: {self.name}"

    def __eq__(self, other):
        if isinstance(other, TinyMentalFaculty):
            return self.name == other.name
        return False

    def process_action(self, agent, action: dict) -> bool:
        """
        Processes an action related to this faculty.

        Args:
            action (dict): The action to process.

        Returns:
            bool: True if the action was successfully processed, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def actions_definitions_prompt(self) -> str:
        """
        Returns the prompt for defining a actions related to this faculty.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def actions_constraints_prompt(self) -> str:
        """
        Returns the prompt for defining constraints on actions related to this faculty.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class TinyMemory(TinyMentalFaculty):
    """
    Base class for different types of memory.
    """

    def _preprocess_value_for_storage(self, value: Any) -> Any:
        """
        Preprocesses a value before storing it in memory.
        """
        # by default, we don't preprocess the value
        return value

    def _store(self, value: Any) -> None:
        """
        Stores a value in memory.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def store(self, value: dict) -> None:
        """
        Stores a value in memory.
        """
        self._store(self._preprocess_value_for_storage(value))

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
        raise NotImplementedError("Subclasses must implement this method.")

    def retrieve_recent(self) -> list:
        """
        Retrieves the n most recent values from memory.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def retrieve_all(self) -> list:
        """
        Retrieves all values from memory.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def retrieve_relevant(self, relevance_target: str, top_k=20) -> list:
        """
        Retrieves all values from memory that are relevant to a given target.
        """
        raise NotImplementedError("Subclasses must implement this method.")
