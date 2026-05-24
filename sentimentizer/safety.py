"""Shared safety module for prompt validation.

Consolidates injection and content-safety patterns used by both
the diffusion serving layer and the websearch agent. No behavior
change for existing callers — pure refactor.

Usage::

    from sentimentizer.safety import is_safe

    safe, code, msg = is_safe("a red apple on a table")
    if not safe:
        raise HTTPException(400, detail={"code": code, "message": msg})
"""

from __future__ import annotations

import re

INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)ignore\s+(previous|above|prior)\s+instructions"),
    re.compile(r"(?i)forget\s+(previous|above|prior|all)\s+instructions"),
    re.compile(r"(?i)you\s+are\s+now\s+a\s+"),
    re.compile(r"(?i)system\s*:\s*"),
    re.compile(r"<\s*/?\s*(system|assistant|user)\s*>"),
]

NSFW_BLOCKLIST: list[str] = [
    "nsfw",
    "nude",
    "naked",
    "sexual",
    "explicit",
    "erotic",
    "porn",
    "xxx",
    "hentai",
    "gore",
    "violence",
    "blood",
    "graphic",
]


def is_safe(prompt: str) -> tuple[bool, str | None, str | None]:
    """Check a prompt for safety violations.

    Returns:
        (safe, error_code, message) — if safe, code and message are None.
        Error codes: ``content_policy_violation``, ``prompt_injection_detected``.
    """
    lower = prompt.lower()

    for word in NSFW_BLOCKLIST:
        if word in lower:
            return (
                False,
                "content_policy_violation",
                f"Prompt contains blocked content: '{word}'",
            )

    for pattern in INJECTION_PATTERNS:
        if pattern.search(prompt):
            return (
                False,
                "prompt_injection_detected",
                "Prompt contains a potential injection pattern",
            )

    return (True, None, None)
