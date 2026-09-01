"""Pytest configuration private to eager speculative-decoding E2E tests."""

import ast

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--test-device-group",
        action="store",
        default="[0]",
        help="QAIC QIDs as a Python list literal, for example '[0,1]'.",
    )


@pytest.fixture(scope="session")
def device_group(request) -> list[int]:
    raw = request.config.getoption("--test-device-group")
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as exc:
        raise pytest.UsageError(
            f"--test-device-group must be a Python list literal, got {raw!r}"
        ) from exc
    if not isinstance(parsed, (list, tuple)):
        raise pytest.UsageError(
            "--test-device-group must parse to a list or tuple, "
            f"got {type(parsed).__name__}"
        )
    return list(parsed)
