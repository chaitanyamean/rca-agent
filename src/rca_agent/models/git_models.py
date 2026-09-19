"""Pydantic models for Git intelligence operations."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ChangeType(str, Enum):
    """How a file was affected by a commit."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    COPIED = "copied"
    UNKNOWN = "unknown"

    @classmethod
    def from_git_status(cls, status: str) -> "ChangeType":
        """Map a single-char git status letter to a ChangeType."""
        mapping: dict[str, ChangeType] = {
            "A": cls.ADDED,
            "M": cls.MODIFIED,
            "D": cls.DELETED,
            "R": cls.RENAMED,
            "C": cls.COPIED,
        }
        return mapping.get(status.upper()[:1], cls.UNKNOWN)


class CommitFile(BaseModel):
    """A single file touched by a commit."""

    file_path: str = Field(description="Repository-relative path of the file.")
    change_type: ChangeType = Field(description="How the file was changed.")
    old_path: str | None = Field(
        default=None,
        description="Previous path for renamed/copied files.",
    )


class CommitDiff(BaseModel):
    """The diff produced by a single file within a commit."""

    commit_id: str = Field(description="Full SHA-1 commit hash.")
    file_path: str = Field(description="Repository-relative path of the file.")
    change_type: ChangeType = Field(description="How the file was changed.")
    additions: int = Field(default=0, description="Number of added lines.")
    deletions: int = Field(default=0, description="Number of deleted lines.")
    patch: str = Field(default="", description="Unified diff patch text.")


class Commit(BaseModel):
    """A Git commit with metadata and the list of changed files."""

    commit_id: str = Field(description="Full 40-character SHA-1 hash.")
    short_id: str = Field(description="Abbreviated 7-character SHA-1.")
    author: str = Field(description="Author display name.")
    author_email: str = Field(description="Author email address.")
    timestamp: datetime = Field(description="Commit author date (UTC).")
    message: str = Field(description="Full commit message.")
    subject: str = Field(description="First line of the commit message.")
    files_changed: list[CommitFile] = Field(
        default_factory=list,
        description="Files touched by this commit.",
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def _ensure_utc(cls, v: Any) -> datetime:
        if isinstance(v, datetime):
            dt = v
        elif isinstance(v, (int, float)):
            dt = datetime.fromtimestamp(float(v), tz=timezone.utc)
        elif isinstance(v, str):
            # Unix timestamp string from git log --format=%at
            try:
                dt = datetime.fromtimestamp(float(v), tz=timezone.utc)
            except ValueError:
                dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        else:
            raise ValueError(f"Cannot parse timestamp: {v!r}")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @property
    def num_files_changed(self) -> int:
        return len(self.files_changed)


class GitCommitQuery(BaseModel):
    """Parameters for searching / filtering commits."""

    keyword: str | None = Field(
        default=None,
        description="Case-insensitive substring match against the commit message.",
    )
    author: str | None = Field(
        default=None,
        description="Case-insensitive substring match against the author name or email.",
    )
    start_time: datetime | None = Field(
        default=None,
        description="Include only commits at or after this time (UTC).",
    )
    end_time: datetime | None = Field(
        default=None,
        description="Include only commits at or before this time (UTC).",
    )
    max_results: int = Field(
        default=100,
        ge=1,
        description="Maximum number of results to return.",
    )

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def _ensure_utc(cls, v: Any) -> datetime | None:
        if v is None:
            return None
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        raise ValueError(f"Expected datetime, got {type(v)}")
