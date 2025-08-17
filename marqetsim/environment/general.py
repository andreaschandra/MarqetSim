"""Environment settings for MarqetSim."""

from datetime import datetime
from pydantic import BaseModel


class Environment(BaseModel):
    """General environment settings for MarqetSim."""

    current_datetime: datetime = datetime.now()
