"""Prompt templates loaded from text files in this directory.

All LLM prompts are stored as .txt files in sentimentizer/prompts/
so they can be easily reviewed and edited without modifying Python code.
"""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Load a prompt template from a text file.

    Args:
        name: Prompt filename (without .txt extension), e.g. "analysis_system".

    Returns:
        The prompt text content.

    Raises:
        FileNotFoundError: If the prompt file doesn't exist.
    """
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text().strip()


# Pre-loaded prompts for backward compatibility
ANALYSIS_SYSTEM_PROMPT = load_prompt("analysis_system")
STRATEGY_SYSTEM_PROMPT = load_prompt("strategy_system")
AUGMENT_ROUTER_PROMPT = load_prompt("augment_router")
ANALYZE_NODE_PROMPT = load_prompt("analyze_node")
DECIDE_NODE_PROMPT = load_prompt("decide_node")

__all__ = [
    "load_prompt",
    "ANALYSIS_SYSTEM_PROMPT",
    "STRATEGY_SYSTEM_PROMPT",
    "AUGMENT_ROUTER_PROMPT",
    "ANALYZE_NODE_PROMPT",
    "DECIDE_NODE_PROMPT",
]
