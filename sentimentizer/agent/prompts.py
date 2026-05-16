"""LLM prompt templates for the tuning agents.

Prompts are loaded from text files in sentimentizer/prompts/
so they can be easily reviewed and edited without modifying Python code.
"""

from sentimentizer.prompts import ANALYSIS_SYSTEM_PROMPT, STRATEGY_SYSTEM_PROMPT

__all__ = ["ANALYSIS_SYSTEM_PROMPT", "STRATEGY_SYSTEM_PROMPT"]
