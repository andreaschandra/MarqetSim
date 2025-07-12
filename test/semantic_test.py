import sys

from marqetsim.agent import SemanticMemory

s_memory = SemanticMemory()


# from person --> listen_and_act
# listen --> observe --> store in memory
# act --> aux_act_once --> _update_cognitive_state --> retrieve_relevant_memories_for_current_context
#     --> retrieve_memories --> episodic_memory.retrieve
#     --> retrieve_relevant_memories --> semantic_memory.retrieve_relevant
