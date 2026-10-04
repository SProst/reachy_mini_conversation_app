"""Check PR metadata without executing its contents."""

import os
import sys
import subprocess
from pathlib import Path

import pytest


CHECKER = Path(__file__).resolve().parents[1] / "scripts" / "check_pr_title.py"


@pytest.mark.parametrize(
    ("title", "expected_status"),
    [
        ("feat: support another language", 0),
        ("fix(audio): retain the selected voice", 0),
        ("feat(api)!: remove the old endpoint", 0),
        ("refactor!: replace the configuration format", 0),
        ("docs: Clarify setup", 0),
        ("ci: validate offline", 0),
        ("build(deps): update the SDK", 0),
        ("revert: undo the endpoint change", 0),
        ("fix: preserve 日本語 text", 0),
        ("", 1),
        ("Add a feature", 1),
        ("feat: ", 1),
        ("feat:  extra leading space", 1),
        ("fix: trailing space ", 1),
        ("fix(): empty scope", 1),
        ("unknown: unsupported type", 1),
        ("FIX: uppercase type", 1),
        ("fix:missing space", 1),
        ("fix: first line\nsecond line", 1),
        ("fix: control\rcharacter", 1),
    ],
)
def test_pr_title_command(title: str, expected_status: int) -> None:
    """Accept supported squash subjects and reject malformed metadata."""
    result = subprocess.run([sys.executable, str(CHECKER), title], capture_output=True, text=True, timeout=10)
    assert result.returncode == expected_status, result.stdout + result.stderr


def test_pr_title_metadata_remains_literal(tmp_path: Path) -> None:
    """Shell syntax in metadata is validated as text without being executed."""
    title = 'fix: $(touch injected) `touch injected` "; touch injected; #'
    result = subprocess.run(
        [sys.executable, str(CHECKER)],
        cwd=tmp_path,
        env={**os.environ, "PR_TITLE": title},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "injected").exists()
