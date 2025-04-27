import logging

import pytest

logger = logging.getLogger("tinytroupe")

import sys

sys.path.append("../../tinytroupe/")
sys.path.append("../../")
sys.path.append("..")


from testing_utils import *

import tinytroupe
import tinytroupe.control as control
from tinytroupe.agent import TinyPerson
from tinytroupe.control import Simulation
from tinytroupe.environment import TinySocialNetwork, TinyWorld
from tinytroupe.examples import (
    create_lisa_the_data_scientist,
    create_marcos_the_physician,
    create_oscar_the_architect,
)
from tinytroupe.extraction import ResultsExtractor
from tinytroupe.extraction import default_extractor as extractor
from tinytroupe.factory import TinyPersonFactory


def test_brainstorming_scenario(setup, focus_group_world):
    world = focus_group_world

    world.broadcast(
        """
             Folks, we need to brainstorm ideas for a new product. Your mission is to discuss potential AI feature ideas
             to add to Microsoft Word. In general, we want features that make you or your industry more productive,
             taking advantage of all the latest AI technologies.

             Please start the discussion now.
             """
    )

    world.run(1)

    agent = TinyPerson.get_agent_by_name("Lisa")

    agent.listen_and_act(
        "Can you please summarize the ideas that the group came up with?"
    )

    from tinytroupe.extraction import ResultsExtractor

    extractor = ResultsExtractor()

    results = extractor.extract_results_from_agent(
        agent,
        extraction_objective="Summarize the the ideas that the group came up with, explaining each idea as an item of a list. Describe in details the benefits and drawbacks of each.",
        situation="A focus group to brainstorm ideas for a new product.",
    )

    print("Brainstorm Results: ", results)

    assert proposition_holds(
        f"The following contains some ideas for new product features or entirely new products: '{results}'"
    ), f"Proposition is false according to the LLM."
