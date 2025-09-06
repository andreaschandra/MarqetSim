"""_summary_"""

from typing import Any

from pydantic import BaseModel


class Action(BaseModel):
    """Action model."""

    type: str
    content: str
    target: str


class CognitiveState(BaseModel):
    """Cognitive state model."""

    goals: str
    attention: str
    emotions: str


class CognitiveActionModel(BaseModel):
    """Cognitive action model."""

    action: Action
    cognitive_state: CognitiveState
