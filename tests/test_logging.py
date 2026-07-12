import logging

from sentimentizer import configured_log_level


def test_configured_log_level_accepts_debug_case_insensitively() -> None:
    assert configured_log_level("debug") == logging.DEBUG


def test_configured_log_level_falls_back_to_info() -> None:
    assert configured_log_level("not-a-level") == logging.INFO
