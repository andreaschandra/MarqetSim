"""MarqetSim is a library for simulating markets and agent interactions."""

from marqetsim import utils

__version__ = "0.1.0"

config = utils.read_config_file(use_cache=True, verbose=True)
