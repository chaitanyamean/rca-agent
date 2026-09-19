"""Safe, controlled Git subprocess wrapper.

Security model
--------------
* ``shell=False`` on every call — arguments are passed as a list, never
  interpolated into a shell string.
* The only Git subcommands allowed are those in ``_ALLOWED_SUBCOMMANDS``.
  Any attempt to call an unlisted subcommand raises ``PermissionError``.
* All operations are read-only subcommands; no write operations are permitted.
* The working directory is validated to be an existing directory before use.
* User-supplied commit IDs are validated against a strict SHA-1 hex pattern
  before they are forwarded to Git, preventing argument injection.

The wrapper intentionally keeps its interface narrow.  Higher-level providers
(``LocalGitProvider``) build domain knowledge on top of it.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Allowlist of Git subcommands this wrapper may invoke.
_ALLOWED_SUBCOMMANDS: frozenset[str] = frozenset(
    {
        "log",
        "show",
        "diff",
        "diff-tree",
        "rev-parse",
        "cat-file",
        "ls-files",
        "name-rev",
    }
)

# Strict pattern: 4–40 hex characters (covers short and full SHA-1).
_COMMIT_ID_RE = re.compile(r"^[0-9a-f]{4,40}$", re.IGNORECASE)

# Timeout for individual git invocations (seconds).
_GIT_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GitError(RuntimeError):
    """Raised when a Git command returns a non-zero exit code."""


class GitNotFoundError(GitError):
    """Raised when a commit, ref, or object is not found in the repository."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_commit_id(commit_id: str) -> str:
    """Return *commit_id* unchanged if it matches the SHA-1 hex pattern.

    Raises
    ------
    ValueError
        If the value does not look like a valid commit ID.
    """
    if not _COMMIT_ID_RE.match(commit_id):
        raise ValueError(
            f"Invalid commit ID {commit_id!r}: must be 4–40 hexadecimal characters."
        )
    return commit_id


def run_git(
    repo_path: Path,
    subcommand: str,
    *args: str,
    check_not_found: bool = True,
) -> str:
    """Execute a single read-only Git command and return stdout as a string.

    Parameters
    ----------
    repo_path:
        Absolute path to the Git repository root.
    subcommand:
        Git subcommand (e.g. ``"log"``, ``"show"``).  Must be in
        ``_ALLOWED_SUBCOMMANDS``.
    *args:
        Additional arguments forwarded verbatim to Git.
    check_not_found:
        When True, a stderr message containing ``"unknown revision"`` or
        ``"bad object"`` is re-raised as ``GitNotFoundError``.

    Raises
    ------
    PermissionError
        If *subcommand* is not in the allowlist.
    ValueError
        If *repo_path* does not exist or is not a directory.
    GitNotFoundError
        If Git cannot resolve the requested object (when *check_not_found*).
    GitError
        For any other non-zero exit code.
    """
    if subcommand not in _ALLOWED_SUBCOMMANDS:
        raise PermissionError(
            f"Git subcommand {subcommand!r} is not in the allowed list."
        )

    if not repo_path.exists():
        raise ValueError(f"Repository path does not exist: {repo_path}")
    if not repo_path.is_dir():
        raise ValueError(f"Repository path is not a directory: {repo_path}")

    cmd: list[str] = ["git", "-C", str(repo_path), subcommand, *args]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            shell=False,  # explicit — never interpolate into a shell string
        )
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"Git command timed out after {_GIT_TIMEOUT}s: {cmd}") from exc
    except FileNotFoundError as exc:
        raise GitError(
            "Git executable not found. Ensure git is installed and on PATH."
        ) from exc

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if check_not_found and _is_not_found_error(stderr):
            raise GitNotFoundError(stderr)
        raise GitError(
            f"git {subcommand} exited with code {result.returncode}: {stderr}"
        )

    return result.stdout


def _is_not_found_error(stderr: str) -> bool:
    """Return True if the stderr text indicates a missing object/revision."""
    not_found_phrases = (
        "unknown revision",
        "bad object",
        "not a valid object name",
        "no such path",
        "ambiguous argument",
        "fatal: bad revision",
    )
    lower = stderr.lower()
    return any(phrase in lower for phrase in not_found_phrases)
