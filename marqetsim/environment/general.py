"""Environment settings for MarqetSim."""

from datetime import datetime


class Environment:
    """General environment settings for MarqetSim."""

    def __init__(self):
        self.current_datetime = datetime.now()
