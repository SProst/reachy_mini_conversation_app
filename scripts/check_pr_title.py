"""Check the Conventional Commit title used for a squash merge."""

import os
import re
import sys
import argparse


TITLE_PATTERN = re.compile(
    r"(?:feat|fix|perf|refactor|docs|test|build|ci|chore|style|revert)"
    r"(?:\([a-z0-9][a-z0-9._/-]*\))?!?: \S(?:.*\S)?"
)


def main() -> int:
    """Validate a literal title from the command line or PR metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("title", nargs="?", default=os.environ.get("PR_TITLE", ""))
    title = parser.parse_args().title
    if not title.isprintable() or TITLE_PATTERN.fullmatch(title) is None:
        print(
            "Use type(scope): description or type(scope)!: description; scope is optional. "
            "Allowed lowercase types: feat, fix, perf, refactor, docs, test, build, ci, chore, style, revert.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
