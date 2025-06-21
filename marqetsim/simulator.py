from marqetsim.agent import Person

def create_person(profile) -> Person:
    person = Person("Budi")
    for k, v in profile.items():
        person.define(k, v)
    return person