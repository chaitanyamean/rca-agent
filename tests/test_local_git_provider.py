"""Tests for LocalGitProvider — Phase 3: Git Intelligence.

A temporary Git repository is created once per test session using only
controlled subprocess calls (the same approach used by the provider itself).
No test depends on any external or pre-existing repository.

Coverage
--------
1.  Retrieve recent commits
2.  Retrieve a specific commit by full and abbreviated SHA
3.  Retrieve diff for a commit
4.  Retrieve files changed in a commit
5.  Search commits by keyword
6.  Filter commits by time range (get_commits_between)
7.  Invalid repository path raises ValueError
8.  Invalid / unknown commit ID raises appropriate error
9.  Commit metadata is correct (author, message, timestamp)
10. Diff contains addition/deletion counts and patch text
11. Security: disallowed git subcommand raises PermissionError
12. Security: invalid commit ID format raises ValueError
13. Acceptance: "What changed in the last 2 hours?"
14. Acceptance: "Show me the diff for commit XYZ."
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rca_agent.models.git_models import ChangeType, GitCommitQuery
from rca_agent.providers.git_subprocess import (
    GitNotFoundError,
    run_git,
    validate_commit_id,
)
from rca_agent.providers.local_git_provider import LocalGitProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git(repo: Path, *args: str) -> str:
    """Run a git command in *repo* and return stdout. Raises on failure."""
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _write(repo: Path, filename: str, content: str) -> Path:
    path = repo / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _commit(repo: Path, message: str) -> str:
    """Stage all changes and create a commit. Returns the full commit SHA."""
    _git(repo, "add", "-A")
    _git(repo, "commit", "--allow-empty", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


# ---------------------------------------------------------------------------
# Session-scoped temp repo fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def git_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a real temporary Git repository with multiple commits.

    Commit layout (oldest → newest):
      C1 — add app/main.py          (feat: initial application scaffold)
      C2 — add app/database.py      (feat: add database connection module)
      C3 — modify app/main.py       (fix: handle startup error gracefully)
      C4 — add app/payments.py      (feat: add payment processor)
      C5 — modify app/payments.py   (fix: fix timeout in payment gateway)
      C6 — modify app/database.py   (fix: fix database connection pool leak)
    """
    repo = tmp_path_factory.mktemp("git_repo")

    # Initialise repo with a known identity so tests are reproducible
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@rca-agent.local")
    _git(repo, "config", "user.name", "RCA Test")

    # C1
    _write(repo, "app/main.py", "def main():\n    pass\n")
    _write(repo, "README.md", "# Test App\n")
    _commit(repo, "feat: initial application scaffold")

    # C2
    _write(repo, "app/database.py", "def connect():\n    return None\n")
    _commit(repo, "feat: add database connection module")

    # C3
    _write(repo, "app/main.py",
           "def main():\n    try:\n        start()\n    except Exception as e:\n        log(e)\n")
    _commit(repo, "fix: handle startup error gracefully")

    # C4
    _write(repo, "app/payments.py",
           "TIMEOUT = 3000\n\ndef process_payment(amount):\n    return {'status': 'ok'}\n")
    _commit(repo, "feat: add payment processor")

    # C5
    _write(repo, "app/payments.py",
           "TIMEOUT = 5000\n\ndef process_payment(amount):\n    if amount <= 0:\n        raise ValueError\n    return {'status': 'ok'}\n")
    _commit(repo, "fix: fix timeout in payment gateway")

    # C6
    _write(repo, "app/database.py",
           "POOL_SIZE = 10\n\ndef connect():\n    return create_pool(POOL_SIZE)\n")
    _commit(repo, "fix: fix database connection pool leak")

    return repo


@pytest.fixture(scope="session")
def provider(git_repo: Path) -> LocalGitProvider:
    """Return a provider backed by the session-scoped temp repo."""
    return LocalGitProvider(git_repo)


# ---------------------------------------------------------------------------
# 1. Retrieve recent commits
# ---------------------------------------------------------------------------

class TestGetRecentCommits:
    def test_returns_list_of_commits(self, provider: LocalGitProvider) -> None:
        commits = provider.get_recent_commits()
        assert isinstance(commits, list)
        assert len(commits) >= 6

    def test_newest_commit_is_first(self, provider: LocalGitProvider) -> None:
        commits = provider.get_recent_commits()
        timestamps = [c.timestamp for c in commits]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_limit_is_respected(self, provider: LocalGitProvider) -> None:
        commits = provider.get_recent_commits(limit=3)
        assert len(commits) == 3

    def test_commits_have_required_fields(self, provider: LocalGitProvider) -> None:
        commit = provider.get_recent_commits(limit=1)[0]
        assert len(commit.commit_id) == 40
        assert len(commit.short_id) <= 7
        assert commit.author
        assert commit.author_email
        assert commit.message
        assert commit.subject
        assert isinstance(commit.timestamp, datetime)
        assert commit.timestamp.tzinfo is not None

    def test_files_changed_populated(self, provider: LocalGitProvider) -> None:
        commits = provider.get_recent_commits(limit=1)
        assert isinstance(commits[0].files_changed, list)
        assert len(commits[0].files_changed) > 0


# ---------------------------------------------------------------------------
# 2. Retrieve a specific commit
# ---------------------------------------------------------------------------

class TestGetCommit:
    def test_get_by_full_sha(self, provider: LocalGitProvider, git_repo: Path) -> None:
        full_sha = _git(git_repo, "rev-parse", "HEAD")
        commit = provider.get_commit(full_sha)
        assert commit.commit_id == full_sha

    def test_get_by_abbreviated_sha(self, provider: LocalGitProvider, git_repo: Path) -> None:
        short_sha = _git(git_repo, "rev-parse", "--short", "HEAD")
        commit = provider.get_commit(short_sha)
        assert commit.commit_id.startswith(short_sha[:4])

    def test_commit_message_matches(self, provider: LocalGitProvider, git_repo: Path) -> None:
        full_sha = _git(git_repo, "rev-parse", "HEAD")
        commit = provider.get_commit(full_sha)
        assert "fix: fix database connection pool leak" in commit.message

    def test_commit_author_is_set(self, provider: LocalGitProvider, git_repo: Path) -> None:
        full_sha = _git(git_repo, "rev-parse", "HEAD")
        commit = provider.get_commit(full_sha)
        assert commit.author == "RCA Test"
        assert commit.author_email == "test@rca-agent.local"

    def test_unknown_commit_raises(self, provider: LocalGitProvider) -> None:
        with pytest.raises((GitNotFoundError, ValueError)):
            provider.get_commit("0" * 40)


# ---------------------------------------------------------------------------
# 3. Retrieve diff
# ---------------------------------------------------------------------------

class TestGetDiff:
    def test_returns_list_of_commit_diffs(
        self, provider: LocalGitProvider, git_repo: Path
    ) -> None:
        sha = _git(git_repo, "rev-parse", "HEAD")
        diffs = provider.get_diff(sha)
        assert isinstance(diffs, list)
        assert len(diffs) > 0

    def test_diff_has_required_fields(
        self, provider: LocalGitProvider, git_repo: Path
    ) -> None:
        sha = _git(git_repo, "rev-parse", "HEAD")
        diff = provider.get_diff(sha)[0]
        assert diff.commit_id == sha
        assert diff.file_path
        assert isinstance(diff.additions, int)
        assert isinstance(diff.deletions, int)
        assert isinstance(diff.patch, str)

    def test_additions_deletions_nonzero_for_modification(
        self, provider: LocalGitProvider, git_repo: Path
    ) -> None:
        # C6 modifies app/database.py — should have both adds and deletes
        sha = _git(git_repo, "rev-parse", "HEAD")
        diffs = provider.get_diff(sha)
        db_diff = next((d for d in diffs if "database.py" in d.file_path), None)
        assert db_diff is not None
        assert db_diff.additions > 0
        assert db_diff.deletions > 0

    def test_patch_contains_diff_markers(
        self, provider: LocalGitProvider, git_repo: Path
    ) -> None:
        sha = _git(git_repo, "rev-parse", "HEAD")
        diffs = provider.get_diff(sha)
        full_patch = "\n".join(d.patch for d in diffs)
        assert "@@" in full_patch

    def test_invalid_commit_raises(self, provider: LocalGitProvider) -> None:
        with pytest.raises((GitNotFoundError, ValueError)):
            provider.get_diff("0" * 40)


# ---------------------------------------------------------------------------
# 4. Retrieve files changed
# ---------------------------------------------------------------------------

class TestGetFilesChanged:
    def test_returns_list_of_strings(
        self, provider: LocalGitProvider, git_repo: Path
    ) -> None:
        sha = _git(git_repo, "rev-parse", "HEAD")
        files = provider.get_files_changed(sha)
        assert isinstance(files, list)
        assert all(isinstance(f, str) for f in files)

    def test_known_file_is_present(
        self, provider: LocalGitProvider, git_repo: Path
    ) -> None:
        # C6 changes app/database.py
        sha = _git(git_repo, "rev-parse", "HEAD")
        files = provider.get_files_changed(sha)
        assert any("database.py" in f for f in files)

    def test_files_match_diff_file_paths(
        self, provider: LocalGitProvider, git_repo: Path
    ) -> None:
        sha = _git(git_repo, "rev-parse", "HEAD")
        files = provider.get_files_changed(sha)
        diffs = provider.get_diff(sha)
        diff_paths = {d.file_path for d in diffs}
        for f in files:
            assert f in diff_paths

    def test_invalid_commit_raises(self, provider: LocalGitProvider) -> None:
        with pytest.raises((GitNotFoundError, ValueError)):
            provider.get_files_changed("0" * 40)


# ---------------------------------------------------------------------------
# 5. Search commits by keyword
# ---------------------------------------------------------------------------

class TestSearchCommits:
    def test_keyword_matches_message(self, provider: LocalGitProvider) -> None:
        query = GitCommitQuery(keyword="payment")
        results = provider.search_commits(query)
        assert len(results) > 0
        assert all("payment" in c.message.lower() for c in results)

    def test_keyword_is_case_insensitive(self, provider: LocalGitProvider) -> None:
        r1 = provider.search_commits(GitCommitQuery(keyword="DATABASE"))
        r2 = provider.search_commits(GitCommitQuery(keyword="database"))
        assert len(r1) == len(r2)
        assert {c.commit_id for c in r1} == {c.commit_id for c in r2}

    def test_keyword_no_match_returns_empty(self, provider: LocalGitProvider) -> None:
        results = provider.search_commits(GitCommitQuery(keyword="zzz-no-match-xyz"))
        assert results == []

    def test_max_results_respected(self, provider: LocalGitProvider) -> None:
        results = provider.search_commits(GitCommitQuery(max_results=2))
        assert len(results) <= 2

    def test_search_by_author(self, provider: LocalGitProvider) -> None:
        results = provider.search_commits(GitCommitQuery(author="RCA Test"))
        assert len(results) > 0
        assert all("rca test" in c.author.lower() for c in results)

    def test_fix_commits_found(self, provider: LocalGitProvider) -> None:
        results = provider.search_commits(GitCommitQuery(keyword="fix:"))
        # We created 3 fix commits in the fixture
        assert len(results) >= 3


# ---------------------------------------------------------------------------
# 6. Filter commits by time range
# ---------------------------------------------------------------------------

class TestGetCommitsBetween:
    def test_returns_commits_within_window(
        self, provider: LocalGitProvider
    ) -> None:
        now = datetime.now(timezone.utc)
        # All fixture commits were just created — a 10-minute window catches them all
        commits = provider.get_commits_between(
            now - timedelta(minutes=10), now
        )
        assert len(commits) >= 6

    def test_future_window_returns_empty(
        self, provider: LocalGitProvider
    ) -> None:
        future = datetime.now(timezone.utc) + timedelta(days=365)
        commits = provider.get_commits_between(future, future + timedelta(hours=1))
        assert commits == []

    def test_past_window_returns_empty(
        self, provider: LocalGitProvider
    ) -> None:
        past_end = datetime.now(timezone.utc) - timedelta(days=365)
        commits = provider.get_commits_between(
            past_end - timedelta(hours=1), past_end
        )
        assert commits == []

    def test_results_sorted_newest_first(
        self, provider: LocalGitProvider
    ) -> None:
        now = datetime.now(timezone.utc)
        commits = provider.get_commits_between(now - timedelta(minutes=10), now)
        if len(commits) > 1:
            timestamps = [c.timestamp for c in commits]
            assert timestamps == sorted(timestamps, reverse=True)

    # Acceptance criteria: "What changed in the last 2 hours?"
    def test_acceptance_what_changed_last_2_hours(
        self, provider: LocalGitProvider
    ) -> None:
        now = datetime.now(timezone.utc)
        commits = provider.get_commits_between(now - timedelta(hours=2), now)
        assert len(commits) >= 6, "Expected all fixture commits within last 2 hours"
        # Each commit should have file information
        for commit in commits:
            assert isinstance(commit.files_changed, list)


# ---------------------------------------------------------------------------
# 7. Invalid repository
# ---------------------------------------------------------------------------

class TestInvalidRepository:
    def test_nonexistent_path_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            LocalGitProvider("/tmp/rca-agent-no-such-repo-xyz")

    def test_non_git_directory_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        # tmp_path exists but has no .git
        with pytest.raises(ValueError, match="not a Git repository"):
            LocalGitProvider(tmp_path)

    def test_file_path_raises_value_error(self, tmp_path: Path) -> None:
        f = tmp_path / "not_a_dir.txt"
        f.write_text("hello")
        with pytest.raises(ValueError):
            LocalGitProvider(f)


# ---------------------------------------------------------------------------
# 8. Invalid commit ID
# ---------------------------------------------------------------------------

class TestInvalidCommitId:
    def test_unknown_full_sha_raises(self, provider: LocalGitProvider) -> None:
        with pytest.raises((GitNotFoundError, ValueError)):
            provider.get_commit("deadbeef" * 5)  # 40-char but unknown

    def test_malformed_id_raises_value_error(self, provider: LocalGitProvider) -> None:
        with pytest.raises(ValueError, match="Invalid commit ID"):
            provider.get_commit("not-a-sha!!")

    def test_too_short_id_raises_value_error(self, provider: LocalGitProvider) -> None:
        # Less than 4 hex chars — below minimum length
        with pytest.raises(ValueError, match="Invalid commit ID"):
            provider.get_commit("ab")

    def test_validate_commit_id_accepts_valid(self) -> None:
        assert validate_commit_id("a1b2c3d") == "a1b2c3d"
        assert validate_commit_id("a" * 40) == "a" * 40

    def test_validate_commit_id_rejects_special_chars(self) -> None:
        with pytest.raises(ValueError):
            validate_commit_id("../../etc/passwd")
        with pytest.raises(ValueError):
            validate_commit_id("abc; rm -rf /")


# ---------------------------------------------------------------------------
# 9. Security: disallowed subcommand
# ---------------------------------------------------------------------------

class TestSecuritySubcommandAllowlist:
    def test_disallowed_subcommand_raises_permission_error(
        self, git_repo: Path
    ) -> None:
        with pytest.raises(PermissionError):
            run_git(git_repo, "push")

    def test_disallowed_commit_raises_permission_error(
        self, git_repo: Path
    ) -> None:
        with pytest.raises(PermissionError):
            run_git(git_repo, "commit", "-m", "injected")

    def test_disallowed_rm_raises_permission_error(
        self, git_repo: Path
    ) -> None:
        with pytest.raises(PermissionError):
            run_git(git_repo, "rm", "README.md")


# ---------------------------------------------------------------------------
# 10. Acceptance: "Show me the diff for commit XYZ."
# ---------------------------------------------------------------------------

class TestAcceptanceDiff:
    def test_get_diff_for_payment_commit(
        self, provider: LocalGitProvider, git_repo: Path
    ) -> None:
        """Acceptance: provider can return the diff for any named commit."""
        # Find the payment gateway fix commit
        results = provider.search_commits(
            GitCommitQuery(keyword="fix: fix timeout in payment gateway")
        )
        assert len(results) >= 1, "Payment gateway fix commit not found"
        commit = results[0]

        diffs = provider.get_diff(commit.commit_id)
        assert len(diffs) > 0

        payment_diff = next((d for d in diffs if "payments.py" in d.file_path), None)
        assert payment_diff is not None, "payments.py diff not found"
        assert payment_diff.additions > 0
        assert "TIMEOUT" in payment_diff.patch or "timeout" in payment_diff.patch.lower()

    def test_diff_commit_id_matches_request(
        self, provider: LocalGitProvider, git_repo: Path
    ) -> None:
        sha = _git(git_repo, "rev-parse", "HEAD")
        diffs = provider.get_diff(sha)
        for diff in diffs:
            assert diff.commit_id == sha
