"""Agent configuration and management."""

import textwrap


class Person:
    def __init__(self, name):
        self.name = name
        self._configuration = {
            "name": self.name,
            "age": None,
            "nationality": None,
            "country_of_residence": None,
            "occupation": None,
            "routines": [],
            "occupation_description": None,
            "personality_traits": [],
            "professional_interests": [],
            "personal_interests": [],
            "skills": [],
            "relationships": [],
        }

    def define(self, key, value):
        """Define Person attributes."""
        if isinstance(value, str):
            if key in self._configuration:
                self._configuration[key] = textwrap.dedent(value)
            else:
                raise ValueError(f"Invalid key: {key}.")
        else:
            if key in self._configuration:
                self._configuration[key] = value
            else:
                raise ValueError(f"Invalid key: {key}.")
