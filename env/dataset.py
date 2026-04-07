"""
PrivRL Dataset — Re-exports from environment.py for backward compatibility.

The canonical dataset now lives inline in environment.py alongside
the tracker taxonomy and reward shaping logic. This module re-exports
the key helpers so that other files (inference.py, app.py) can still
import from env.dataset without breaking.
"""

from env.environment import (
    TASKS,
    ALL_TASK_IDS,
    EASY_SITES,
    MEDIUM_SITES,
    HARD_SITES,
    TRACKER_CATEGORIES,
    TRACKER_RISK_WEIGHT,
    classify_tracker,
    compute_tracker_risk,
)


# Combined flat list of all sites across all tiers
ALL_SITES = EASY_SITES + MEDIUM_SITES + HARD_SITES


def get_sites_for_task(task_id: str) -> list:
    """Return the list of website dicts for a given task difficulty."""
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id '{task_id}'. Must be one of: {ALL_TASK_IDS}")
    return TASKS[task_id]["sites"]


def get_all_task_ids() -> list:
    """Return all available task IDs."""
    return list(ALL_TASK_IDS)
