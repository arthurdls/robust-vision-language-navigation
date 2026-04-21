"""
Centralized default configuration for the rvln project.

Model names and numeric defaults used across the codebase live here
so they can be changed in one place.
"""

# LLM model defaults
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_VLM_MODEL = "gpt-5.4"

# Diary monitoring defaults
DEFAULT_MAX_STEPS_PER_SUBGOAL = 300
DEFAULT_DIARY_CHECK_INTERVAL = 10
DEFAULT_MAX_CORRECTIONS = 15
DEFAULT_STALL_WINDOW = 3
DEFAULT_STALL_THRESHOLD = 0.05
DEFAULT_STALL_COMPLETION_FLOOR = 0.8
