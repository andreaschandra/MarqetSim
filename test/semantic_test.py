import os

from marqetsim.agent import SemanticMemory

BASED_PATH = os.path.join("test", "document_test")
document_titles = ["doc1.txt", "doc2.txt"]
all_doc_paths = [os.path.join(BASED_PATH, f) for f in document_titles]
s_memory = SemanticMemory(documents_paths=all_doc_paths)


# from person --> listen_and_act
# listen --> observe --> store in memory
# act --> aux_act_once --> _update_cognitive_state
#     --> retrieve_relevant_memories_for_current_context (set current context from episodic then use that for get relevant memories from semantic, use both for reset prompt)
#     --> retrieve_memories --> episodic_memory.retrieve
#     --> retrieve_relevant_memories --> semantic_memory.retrieve_relevant
#     --> reset_prompt (use self._configuration which contains all the memories) --> generate_agent_system_prompt
#     -->

# example relevance target for semantic:
# "\nCurrent Context: You decided you want to visit Europe and you are planning your next vacations.
# You start by searching for good deals as well as good ideas.\n
# Current Goals: Analyze travel package advertisements\n
# Current Attention: Comparing different European tour options\n
# Current Emotions: Curious and analytical\n
# Recent Memories:\n
# - Info: there were other messages here, but they were omitted for brevity.\n
# - Info: there were other messages here, but they were omitted for brevity.\n
# - Info: there were other messages here, but they were omitted for brevity.\n
# - {'stimuli': [{'type': 'CONVERSATION',
# 'content': 'Can you please evaluate these Bing ads for me? Which one convices you more to buy their particular o (...)',
# 'source': ''}]}\n
# - {'action': {'type': 'THINK',
# 'content': \"As a data manager from Singapore, I'm interested in travel options.
# I'll evaluate these ads objectiv (...)\", 'target': ''},
# 'cognitive_state': {'goals': 'Analyze travel package advertisements',
# 'attention': 'Comparing different European tour options', 'emotions': 'Curious and analytical'}}\n        "


# context = self._configuration["current_context"] --> from input
#         goals = self._configuration["current_goals"] --> produce message
#         attention = self._configuration["current_attention"] --> produce message
#         emotions = self._configuration["current_emotions"] --> produce message
