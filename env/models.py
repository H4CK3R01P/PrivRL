"""
PrivRL Models — Type-safe data contracts for the privacy risk classification environment.

Defines the Action, Observation, and State models following the OpenEnv spec.
These Pydantic models serve as the API contract between client and server.
"""

from typing import List, Optional
from pydantic import BaseModel


# =============================================================================
# Action Model
# =============================================================================

class PrivRLAction(BaseModel):
    """
    An agent's classification decision for a website's privacy risk.

    Valid actions:
        - "mark_safe"      → Website has low privacy risk
        - "mark_risky"     → Website has moderate privacy risk
        - "mark_dangerous" → Website has high privacy risk
    """
    classification: str  # One of: "mark_safe", "mark_risky", "mark_dangerous"


# =============================================================================
# Observation Model
# =============================================================================

class PrivRLObservation(BaseModel):
    """
    The observable state of a simulated website presented to the agent.

    Fields:
        cookies:        Number of cookies the website sets (int, 0+)
        trackers:       List of third-party tracker domains found on the site
        https:          Whether the site uses HTTPS (True) or HTTP (False)
        privacy_policy: Text snippet from the website's privacy policy
        task_id:        Identifier for the current task (easy/medium/hard)
        site_name:      Human-readable name of the simulated website
        done:           Whether the episode is finished
        reward:         Reward from the last action (None on reset)
        message:        Feedback message to the agent
    """
    cookies: int
    trackers: List[str]
    https: bool
    privacy_policy: str
    task_id: str
    site_name: str
    done: bool = False
    reward: Optional[float] = None
    message: str = ""


# =============================================================================
# State Model (server-side, includes ground truth)
# =============================================================================

class PrivRLState(BaseModel):
    """
    Internal episode state — includes fields NOT shown to the agent (ground truth).

    Fields:
        episode_id:     Unique identifier for this episode
        step_count:     How many steps have been taken
        current_index:  Index into the dataset for the current website
        task_id:        Current task difficulty level
        ground_truth:   The correct classification label (hidden from agent)
        total_reward:   Cumulative reward for this episode
        total_sites:    Total number of websites to classify in this episode
        sites_done:     Number of websites classified so far
    """
    episode_id: Optional[str] = None
    step_count: int = 0
    current_index: int = 0
    task_id: str = ""
    ground_truth: str = ""
    total_reward: float = 0.0
    total_sites: int = 0
    sites_done: int = 0
