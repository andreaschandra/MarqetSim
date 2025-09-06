"""Preset of agents for the MarqetSim simulation."""

import os
import sys

from marqetsim.agent.person import Person
from marqetsim.utils.logger import LogCreator
from marqetsim.config import read_config_file

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def create_joe_the_analyst(settings, logger=None):
    """Create a Joe the Analyst agent."""
    person = Person("Joe", settings, logger=logger)
    person.define("age", "35")
    person.define("nationality", "American")
    person.define("country_of_residence", "USA")
    person.define("occupation", "Data Analyst")
    person.define(
        "routines",
        "Joe's daily routine includes data cleaning, analysis, and reporting.",
    )
    person.define(
        "personality_traits",
        [
            {
                "trait": """Joe is analytical, detail-oriented, and enjoys problem-solving. He is also a good 
                communicator and works well in teams."""
            },
            {
                "trait": """Joe is curious and enjoys learning new data analysis techniques and tools. He is also open 
                to feedback and continuously seeks to improve his skills."""
            },
        ],
    )
    person.define(
        "professional_interests",
        [
            {
                "interest": """Joe is interested in machine learning, data visualization, and business intelligence. 
                He enjoys working with data to uncover trends and patterns."""
            },
            {
                "interest": """Joe is also interested in data ethics and responsible AI practices. He believes in using 
                data for good and ensuring that data-driven decisions are fair and transparent."""
            },
        ],
    )
    person.define(
        "personal_interests",
        [
            {
                "interest": """Joe enjoys hiking, photography, and playing chess. He often combines his love for nature 
                with his passion for data by analyzing environmental data."""
            },
            {
                "interest": """Joe is also a fan of science fiction literature and enjoys exploring the intersection of 
                technology and society."""
            },
        ],
    )
    person.define(
        "skills",
        [
            {
                "skill": """Joe is proficient in Python, R, and SQL. He has experience with data visualization tools 
                like Tableau and Power BI."""
            },
            {
                "skill": """Joe is also skilled in statistical analysis and has a strong understanding of data cleaning 
                and preprocessing techniques."""
            },
        ],
    )
    return person


if __name__ == "__main__":
    logger = LogCreator("examples", level="DEBUG")
    settings = read_config_file(logger=logger)

    # Example usage
    joe = create_joe_the_analyst(logger=logger)
    # logger.info(f"Created agent: {joe._configuration}") --> should not be accessed

    joe.set_context(
        """
        You decided you want to visit Europe and you are planning your next vacations. You start by searching for good 
        deals as well as good ideas.
        """
    )

    joe.listen_and_act(
        """Can you please evaluate these Bing ads for me? Which one convices you more to buy their particular offering? 
        Select **ONLY** one. Please explain your reasoning, based on your background and personality.

        Options:
        1. **Visit Paris**: Experience the romance of Paris with a 3-day package including flights and hotel for $499.
        2. **Explore Rome**: Discover the history of Rome with a 5-day tour including flights, hotel, and guided 
        tours for $799.
        3. **Relax in Barcelona**: Enjoy the sun and culture of Barcelona with a 4-day package including flights 
        and hotel for $599.
        """
    )
