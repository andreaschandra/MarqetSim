from tinytroupe.agent import TinyPerson
from tinytroupe.examples import (
    create_lisa_the_data_scientist,
    create_marcos_the_physician,
    create_oscar_the_architect,
)

agent1 = (
    TinyPerson.all_agents["Lisa"]
    if TinyPerson.has_agent("Lisa")
    else create_lisa_the_data_scientist()
)

agent2 = (
    TinyPerson.all_agents["Marcos"]
    if TinyPerson.has_agent("Marcos")
    else create_marcos_the_physician()
)

agent3 = (
    TinyPerson.all_agents["Joko"]
    if TinyPerson.has_agent("Joko")
    else create_oscar_the_architect()
)

agent3.define("name", "Joko")
agent3.define("age", 28)
agent3.define("nationality", "Indonesian")

agent4 = (
    TinyPerson.all_agents["Oscar"]
    if TinyPerson.has_agent("Oscar")
    else create_oscar_the_architect()
)

print(agent)
