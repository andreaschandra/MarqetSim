"""simulator functions"""

from marqetsim.agent import Person


def create_person(profile) -> Person:
    """Create a person from the given profile"""
    person = Person("Budi")
    for k, v in profile.items():
        person.define(k, v)
    return person
