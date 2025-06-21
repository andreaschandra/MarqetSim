"""MarqetSim is a library for simulating markets and agent interactions."""

from marqetsim.utils import common

__version__ = "0.1.1"

config = common.read_config_file(use_cache=True, verbose=True)
