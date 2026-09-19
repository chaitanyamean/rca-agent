"""LocalGitProvider — read-only Git intelligence via controlled subprocess calls.

All user-supplied commit IDs are validated against a strict SHA-1 hex pattern
before being forwarded to Git.  No write operations are exposed.  The working
directory is resolved and validated at construction time.

Usage::

    from rca_agent.providers.local_git_provider import LocalGitProvider

    provider = LocalGitProvider("/path/to/any/repo")

    # What changed in the last 2 hours?
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    commits = provider.get_commits_between(now - timedelta(hours=2), now)

    # Show the diff for a specific commit
    diffs = provider.get_diff("a1b2c3d")
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from rca_agent.models.git_models import (
    ChangeType,
    Commit,
    CommitDiff,
    CommitFile,
    GitCommitQuery,
)
from rca_agent.providers.git_subprocess import (
    GitError,
    GitNotFoundError,
    run_git,
    validate_commit_id,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Format string used for every `git log` call.
# Fields are separated by US (ASCII 0x1F) and records by RS (ASCII 0x1E)
# so they survive any content in the commit message.
# ---------------------------------------------------------------------------
_US = "\x1f"   # Unit Separator
_RS = "\x1e"   # Record Separator

# git log --format fields: hash, short-hash, author-name, author-email,
#                           unix-timestamp, raw message
_LOG_FORMAT = f"%H{_US}%h{_US}%an{_US}%ae{_US}%at{_US}%B{_RS}"


class LocalGitProvider:
    """Read-only Git intelligence provider backed by a local repository.

    Parameters
    ----------
    repo_path:
        Path to the root of a Git repository.  May be absolute or relative.
    max_commits:
        Hard cap on commits loaded by ``get_recent_commits()``.
    """

    def __init__(self, repo_path: str | Path, max_commits: int = 500) -> None:
        self._repo_path = Path(repo_path).resolve()
        self._max_commits = max_commits

        if not self._repo_path.exists():
            raise ValueError(
                f"LocalGitProvider: repository path does not exist: {self._repo_path}"
            )
        if not self._repo_path.is_dir():
            raise ValueError(
                f"LocalGitProvider: repository path is not a directory: {self._repo_path}"
            )

        # Verify this actually is a git repository
        try:
            run_git(self._repo_path, "rev-parse", "--git-dir")
        except GitError as exc:
            raise ValueError(
                f"LocalGitProvider: {self._repo_path} is not a Git repository: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_recent_commits(self, limit: int = 20) -> list[Commit]:
        """Return the *limit* most recent commits, newest first."""
        effective_limit = min(limit, self._max_commits)
        raw = run_git(
            self._repo_path,
            "log",
            f"--max-count={effective_limit}",
            f"--format={_LOG_FORMAT}",
        )
        commits = self._parse_log_output(raw)
        for commit in commits:
            commit.files_changed = self._get_commit_files(commit.commit_id)
        return commits

    def get_commit(self, commit_id: str) -> Commit:
        """Return a single commit by full or abbreviated SHA-1.

        Raises
        ------
        ValueError
            If *commit_id* is syntactically invalid.
        GitNotFoundError
            If the commit does not exist in the repository.
        """
        safe_id = validate_commit_id(commit_id)
        raw = run_git(
            self._repo_path,
            "log",
            "--max-count=1",
            f"--format={_LOG_FORMAT}",
            safe_id,
        )
        commits = self._parse_log_output(raw)
        if not commits:
            raise GitNotFoundError(f"Commit not found: {commit_id!r}")
        commit = commits[0]
        commit.files_changed = self._get_commit_files(commit.commit_id)
        return commit

    def get_diff(self, commit_id: str) -> list[CommitDiff]:
        """Return per-file unified diffs for the given commit.

        Raises
        ------
        ValueError
            If *commit_id* is syntactically invalid.
        GitNotFoundError
            If the commit does not exist in the repository.
        """
        safe_id = validate_commit_id(commit_id)
        # Resolve to full hash first (validates existence)
        full_id = self._resolve_commit(safe_id)

        # --stat gives us addition/deletion counts; -p gives the patch
        raw = run_git(
            self._repo_path,
            "show",
            "--format=",         # suppress commit header
            "--patch",
            "--unified=3",
            full_id,
        )
        return self._parse_show_output(full_id, raw)

    def get_files_changed(self, commit_id: str) -> list[str]:
        """Return repository-relative file paths changed in *commit_id*.

        Raises
        ------
        ValueError
            If *commit_id* is syntactically invalid.
        GitNotFoundError
            If the commit does not exist in the repository.
        """
        safe_id = validate_commit_id(commit_id)
        full_id = self._resolve_commit(safe_id)
        files = self._get_commit_files(full_id)
        return [f.file_path for f in files]

    def search_commits(self, query: GitCommitQuery) -> list[Commit]:
        """Return commits matching the criteria in *query*.

        Filtering is performed in Python after fetching a broad candidate set
        from Git so all filter combinations work without complex JQL.
        """
        args: list[str] = [
            f"--max-count={query.max_results * 4}",  # over-fetch to allow filtering
            f"--format={_LOG_FORMAT}",
        ]
        if query.start_time:
            args += ["--after", _git_date(query.start_time)]
        if query.end_time:
            args += ["--before", _git_date(query.end_time)]
        if query.keyword:
            args += ["--grep", query.keyword, "--regexp-ignore-case"]
        if query.author:
            args += ["--author", query.author]

        raw = run_git(self._repo_path, "log", *args)
        commits = self._parse_log_output(raw)

        # Post-filter (handles cases git --grep doesn't cover, e.g. author email)
        if query.author:
            a = query.author.lower()
            commits = [
                c for c in commits
                if a in c.author.lower() or a in c.author_email.lower()
            ]

        # Trim to requested max
        commits = commits[: query.max_results]

        # Attach file lists
        for commit in commits:
            commit.files_changed = self._get_commit_files(commit.commit_id)

        return commits

    def get_commits_between(
        self, start_time: datetime, end_time: datetime
    ) -> list[Commit]:
        """Return commits whose author date falls within [start_time, end_time].

        Both datetimes are interpreted as UTC if naive.
        """
        start = _ensure_utc(start_time)
        end = _ensure_utc(end_time)

        raw = run_git(
            self._repo_path,
            "log",
            f"--max-count={self._max_commits}",
            f"--format={_LOG_FORMAT}",
            "--after", _git_date(start),
            "--before", _git_date(end),
        )
        commits = self._parse_log_output(raw)
        for commit in commits:
            commit.files_changed = self._get_commit_files(commit.commit_id)
        return commits

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_commit(self, commit_id: str) -> str:
        """Resolve an abbreviated commit ID to its full 40-char SHA-1.

        Uses ``cat-file -e`` to verify the object actually exists in the
        repository (``rev-parse --verify`` can silently pass syntactically
        valid but non-existent SHAs on some Git versions).
        """
        try:
            full = run_git(
                self._repo_path, "rev-parse", "--verify", commit_id
            ).strip()
        except GitError as exc:
            raise GitNotFoundError(f"Commit not found: {commit_id!r}") from exc

        # Confirm the object really exists in the object store
        try:
            run_git(self._repo_path, "cat-file", "-e", full)
        except GitError as exc:
            raise GitNotFoundError(
                f"Commit object does not exist in repository: {commit_id!r}"
            ) from exc

        return full

    def _get_commit_files(self, full_commit_id: str) -> list[CommitFile]:
        """Return CommitFile list for a commit using diff-tree."""
        try:
            raw = run_git(
                self._repo_path,
                "diff-tree",
                "--no-commit-id",
                "-r",
                "--name-status",
                full_commit_id,
            )
        except GitError:
            logger.debug("Could not get file list for %s", full_commit_id)
            return []

        files: list[CommitFile] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", maxsplit=2)
            if len(parts) < 2:
                continue
            status = parts[0][:1]
            change_type = ChangeType.from_git_status(status)

            if change_type in (ChangeType.RENAMED, ChangeType.COPIED) and len(parts) == 3:
                old_path, new_path = parts[1], parts[2]
                files.append(
                    CommitFile(
                        file_path=new_path,
                        change_type=change_type,
                        old_path=old_path,
                    )
                )
            else:
                files.append(
                    CommitFile(file_path=parts[1], change_type=change_type)
                )
        return files

    def _parse_log_output(self, raw: str) -> list[Commit]:
        """Parse `git log --format=_LOG_FORMAT` output into Commit objects."""
        commits: list[Commit] = []
        # Records are separated by RS (\x1e)
        for record in raw.split(_RS):
            record = record.strip()
            if not record:
                continue
            parts = record.split(_US, maxsplit=5)
            if len(parts) < 6:
                logger.debug("Skipping malformed log record: %r", record[:80])
                continue
            commit_id, short_id, author, author_email, ts, message = parts
            subject = message.strip().splitlines()[0] if message.strip() else ""
            try:
                commits.append(
                    Commit(
                        commit_id=commit_id.strip(),
                        short_id=short_id.strip(),
                        author=author.strip(),
                        author_email=author_email.strip(),
                        timestamp=ts.strip(),
                        message=message.strip(),
                        subject=subject,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping unparseable commit record: %s", exc)

        return commits

    def _parse_show_output(self, commit_id: str, raw: str) -> list[CommitDiff]:
        """Parse `git show --patch` output into a list of CommitDiff objects."""
        diffs: list[CommitDiff] = []
        current_file: str | None = None
        current_patch_lines: list[str] = []
        additions = 0
        deletions = 0
        change_type = ChangeType.UNKNOWN

        def _flush() -> None:
            if current_file is not None:
                diffs.append(
                    CommitDiff(
                        commit_id=commit_id,
                        file_path=current_file,
                        change_type=change_type,
                        additions=additions,
                        deletions=deletions,
                        patch="\n".join(current_patch_lines),
                    )
                )

        for line in raw.splitlines():
            if line.startswith("diff --git "):
                _flush()
                current_file = None
                current_patch_lines = [line]
                additions = 0
                deletions = 0
                change_type = ChangeType.UNKNOWN
            elif line.startswith("--- ") or line.startswith("+++ "):
                current_patch_lines.append(line)
                # Detect file path from +++ line
                if line.startswith("+++ b/"):
                    current_file = line[6:]
                    change_type = ChangeType.MODIFIED
                elif line.startswith("+++ /dev/null"):
                    change_type = ChangeType.DELETED
            elif line.startswith("new file mode"):
                change_type = ChangeType.ADDED
                current_patch_lines.append(line)
            elif line.startswith("deleted file mode"):
                change_type = ChangeType.DELETED
                current_patch_lines.append(line)
            elif line.startswith("rename to "):
                change_type = ChangeType.RENAMED
                current_file = line[len("rename to "):]
                current_patch_lines.append(line)
            elif current_file is not None:
                if line.startswith("+") and not line.startswith("+++"):
                    additions += 1
                elif line.startswith("-") and not line.startswith("---"):
                    deletions += 1
                current_patch_lines.append(line)
            else:
                if current_patch_lines:
                    current_patch_lines.append(line)

        _flush()
        return diffs


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _git_date(dt: datetime) -> str:
    """Format a datetime as an ISO-8601 string that git --after/--before accepts."""
    return _ensure_utc(dt).strftime("%Y-%m-%dT%H:%M:%SZ")
