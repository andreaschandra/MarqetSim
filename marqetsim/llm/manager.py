"""LLM Manager handle all provider"""

from marqetsim.llm import LLMBase


def get_llm(settings):
    """
    Retrieves the LLM (Large Language Model) class based on the provided settings.
    Args:
        settings (dict): A dictionary containing configuration settings. Must include
            a "Simulation" key with an "LLM_TYPE" specifying the desired LLM class name.
        logger (Logger, optional): Logger instance for logging purposes. Defaults to LogCreator().
    Returns:
        type: The LLM class corresponding to the specified "LLM_TYPE".
    Raises:
        ValueError: If the specified "LLM_TYPE" is not among the available LLM classes.
    """

    llm_name = settings["Simulation"].get("LLM_TYPE")
    avail_llms = [cls.__name__ for cls in LLMBase.__subclasses__()]

    if llm_name not in avail_llms:
        raise ValueError(
            f"LLM_TYPE '{llm_name}' is not among the available LLMs: {avail_llms}"
        )

    for cls in LLMBase.__subclasses__():
        if cls.__name__ == llm_name:
            return cls(settings)
