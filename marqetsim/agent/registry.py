"""simulator functions"""

from .person import Person


def create_person(profile, logger=None) -> Person:
    """Create a person from the given profile"""
    person = Person(profile.get("name", "Fulan"), logger=logger)
    for k, v in profile.items():
        person.define(k, v)
    return person
