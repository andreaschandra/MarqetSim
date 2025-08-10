"""Environment settings for MarqetSim."""

from datetime import datetime


class Environment:
    """General environment settings for MarqetSim."""

    def __init__(self):
        self.world = {}
        self.world["current_datetime"] = datetime.now()

    def set_param(self, key, value):
        """Set a parameter in the environment."""
        self.world[key] = value

    def get_param(self, key):
        """Get a parameter from the environment."""
        return self.world.get(key, None)
