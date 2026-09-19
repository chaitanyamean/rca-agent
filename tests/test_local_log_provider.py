"""Tests for LocalLogProvider — Phase 2: Structured Log Provider.

Coverage
--------
1.  Search by time range
2.  Search by service
3.  Search by level
4.  Search by trace ID (via search_logs)
5.  Search by endpoint
6.  Search by keyword
7.  Combined filters
8.  Malformed JSON lines are skipped; valid lines still returned
9.  Empty results (no match)
10. get_logs_by_trace_id convenience method
11. get_log_by_id — found and not-found cases
12. Results are sorted ascending by timestamp
13. Instantiation with a non-existent path raises FileNotFoundError
14. LogSearchQuery rejects inverted time range
15. Acceptance-criteria test: service="payment-service", level="ERROR"
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from rca_agent.models.log_entry import LogEntry, LogSearchQuery
from rca_agent.providers.local_log_provider import LocalLogProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc(iso: str) -> datetime:
    """Parse an ISO-8601 string and return a UTC-aware datetime."""
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# 1. Search by time range
# ---------------------------------------------------------------------------

class TestSearchByTime:
    def test_start_time_excludes_earlier_entries(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        query = LogSearchQuery(start_time=_utc("2026-09-19T10:10:00.000Z"))
        result = sample_log_provider.search_logs(query)
        assert all(e.timestamp >= _utc("2026-09-19T10:10:00.000Z") for e in result.entries)
        assert result.total == len(result.entries)

    def test_end_time_excludes_later_entries(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        query = LogSearchQuery(end_time=_utc("2026-09-19T10:05:00.000Z"))
        result = sample_log_provider.search_logs(query)
        assert all(e.timestamp <= _utc("2026-09-19T10:05:00.000Z") for e in result.entries)

    def test_time_window_returns_only_window_entries(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        start = _utc("2026-09-19T10:05:00.000Z")
        end = _utc("2026-09-19T10:08:00.000Z")
        query = LogSearchQuery(start_time=start, end_time=end)
        result = sample_log_provider.search_logs(query)
        assert result.total > 0
        for entry in result.entries:
            assert start <= entry.timestamp <= end

    def test_no_entries_outside_future_window(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        query = LogSearchQuery(start_time=_utc("2030-01-01T00:00:00.000Z"))
        result = sample_log_provider.search_logs(query)
        assert result.total == 0
        assert result.entries == []


# ---------------------------------------------------------------------------
# 2. Search by service
# ---------------------------------------------------------------------------

class TestSearchByService:
    def test_returns_only_matching_service(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        query = LogSearchQuery(service="payment-service")
        result = sample_log_provider.search_logs(query)
        assert result.total > 0
        assert all(e.service == "payment-service" for e in result.entries)

    def test_different_service_returns_different_entries(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        r1 = sample_log_provider.search_logs(LogSearchQuery(service="rke-backend"))
        r2 = sample_log_provider.search_logs(LogSearchQuery(service="auth-service"))
        ids1 = {e.id for e in r1.entries}
        ids2 = {e.id for e in r2.entries}
        assert ids1.isdisjoint(ids2)

    def test_unknown_service_returns_empty(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        query = LogSearchQuery(service="nonexistent-svc")
        result = sample_log_provider.search_logs(query)
        assert result.total == 0


# ---------------------------------------------------------------------------
# 3. Search by level
# ---------------------------------------------------------------------------

class TestSearchByLevel:
    def test_error_level_only_returns_errors(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        query = LogSearchQuery(level="ERROR")
        result = sample_log_provider.search_logs(query)
        assert result.total > 0
        assert all(e.level == "ERROR" for e in result.entries)

    def test_warn_level_only_returns_warnings(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        query = LogSearchQuery(level="WARN")
        result = sample_log_provider.search_logs(query)
        assert result.total > 0
        assert all(e.level == "WARN" for e in result.entries)

    def test_info_level_only_returns_info(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        query = LogSearchQuery(level="INFO")
        result = sample_log_provider.search_logs(query)
        assert result.total > 0
        assert all(e.level == "INFO" for e in result.entries)

    def test_level_filter_is_case_insensitive(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        r_upper = sample_log_provider.search_logs(LogSearchQuery(level="ERROR"))
        r_lower = sample_log_provider.search_logs(LogSearchQuery(level="error"))
        assert r_upper.total == r_lower.total
        assert {e.id for e in r_upper.entries} == {e.id for e in r_lower.entries}


# ---------------------------------------------------------------------------
# 4. Search by trace ID (via search_logs)
# ---------------------------------------------------------------------------

class TestSearchByTraceId:
    def test_returns_only_entries_with_matching_trace_id(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        query = LogSearchQuery(trace_id="trace-005")
        result = sample_log_provider.search_logs(query)
        assert result.total == 2  # two entries share trace-005 in the fixture
        assert all(e.trace_id == "trace-005" for e in result.entries)

    def test_unknown_trace_id_returns_empty(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        query = LogSearchQuery(trace_id="trace-does-not-exist")
        result = sample_log_provider.search_logs(query)
        assert result.total == 0


# ---------------------------------------------------------------------------
# 5. Search by endpoint
# ---------------------------------------------------------------------------

class TestSearchByEndpoint:
    def test_returns_only_matching_endpoint(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        query = LogSearchQuery(endpoint="/api/payments")
        result = sample_log_provider.search_logs(query)
        assert result.total > 0
        assert all(e.endpoint == "/api/payments" for e in result.entries)

    def test_different_endpoint_not_included(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        query = LogSearchQuery(endpoint="/api/payments")
        result = sample_log_provider.search_logs(query)
        assert all(e.endpoint != "/api/orders" for e in result.entries)

    def test_unknown_endpoint_returns_empty(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        result = sample_log_provider.search_logs(
            LogSearchQuery(endpoint="/nonexistent/path")
        )
        assert result.total == 0


# ---------------------------------------------------------------------------
# 6. Search by keyword
# ---------------------------------------------------------------------------

class TestSearchByKeyword:
    def test_keyword_matches_substring_in_message(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        query = LogSearchQuery(keyword="Database connection failed")
        result = sample_log_provider.search_logs(query)
        assert result.total > 0
        assert all("database connection failed" in e.message.lower() for e in result.entries)

    def test_keyword_search_is_case_insensitive(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        r1 = sample_log_provider.search_logs(LogSearchQuery(keyword="timeout"))
        r2 = sample_log_provider.search_logs(LogSearchQuery(keyword="TIMEOUT"))
        assert r1.total == r2.total
        assert {e.id for e in r1.entries} == {e.id for e in r2.entries}

    def test_keyword_with_no_matches_returns_empty(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        result = sample_log_provider.search_logs(
            LogSearchQuery(keyword="zzz-unlikely-keyword-xyz")
        )
        assert result.total == 0


# ---------------------------------------------------------------------------
# 7. Combined filters
# ---------------------------------------------------------------------------

class TestCombinedFilters:
    def test_service_and_level_combined(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        """Acceptance criteria: service='payment-service', level='ERROR'."""
        query = LogSearchQuery(service="payment-service", level="ERROR")
        result = sample_log_provider.search_logs(query)
        assert result.total > 0
        assert all(e.service == "payment-service" for e in result.entries)
        assert all(e.level == "ERROR" for e in result.entries)

    def test_service_and_endpoint_combined(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        query = LogSearchQuery(service="rke-backend", endpoint="/api/orders")
        result = sample_log_provider.search_logs(query)
        assert result.total > 0
        assert all(e.service == "rke-backend" and e.endpoint == "/api/orders"
                   for e in result.entries)

    def test_level_and_keyword_combined(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        query = LogSearchQuery(level="ERROR", keyword="timeout")
        result = sample_log_provider.search_logs(query)
        assert result.total > 0
        assert all(e.level == "ERROR" for e in result.entries)
        assert all("timeout" in e.message.lower() for e in result.entries)

    def test_time_range_and_service_combined(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        query = LogSearchQuery(
            service="api-gateway",
            start_time=_utc("2026-09-19T10:00:00.000Z"),
            end_time=_utc("2026-09-19T10:05:00.000Z"),
        )
        result = sample_log_provider.search_logs(query)
        assert result.total > 0
        for entry in result.entries:
            assert entry.service == "api-gateway"
            assert _utc("2026-09-19T10:00:00.000Z") <= entry.timestamp <= _utc("2026-09-19T10:05:00.000Z")

    def test_all_filters_combined_no_match(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        # Contradictory: payment-service does not have INFO logs on /api/orders
        query = LogSearchQuery(
            service="payment-service",
            level="INFO",
            endpoint="/api/orders",
        )
        result = sample_log_provider.search_logs(query)
        assert result.total == 0


# ---------------------------------------------------------------------------
# 8. Malformed JSON lines are skipped
# ---------------------------------------------------------------------------

class TestMalformedLines:
    def test_valid_entries_returned_despite_bad_lines(
        self, malformed_log_provider: LocalLogProvider
    ) -> None:
        """Provider must not raise; valid lines are still returned."""
        result = malformed_log_provider.search_logs(LogSearchQuery())
        # 3 valid lines in malformed_logs.jsonl
        assert result.total == 3

    def test_error_lines_still_findable(
        self, malformed_log_provider: LocalLogProvider
    ) -> None:
        result = malformed_log_provider.search_logs(LogSearchQuery(level="ERROR"))
        assert result.total == 1
        assert result.entries[0].exception == "RuntimeError"

    def test_from_raw_line_raises_on_bad_json(self) -> None:
        with pytest.raises(ValueError, match="Invalid JSON"):
            LogEntry.from_raw_line("this is not json")

    def test_from_raw_line_raises_on_truncated_json(self) -> None:
        with pytest.raises(ValueError):
            LogEntry.from_raw_line("{broken json here")


# ---------------------------------------------------------------------------
# 9. Empty results
# ---------------------------------------------------------------------------

class TestEmptyResults:
    def test_no_entries_for_unknown_service(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        result = sample_log_provider.search_logs(
            LogSearchQuery(service="ghost-service")
        )
        assert result.total == 0
        assert result.entries == []

    def test_total_matches_entries_length_when_empty(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        result = sample_log_provider.search_logs(
            LogSearchQuery(service="ghost-service")
        )
        assert result.total == len(result.entries)


# ---------------------------------------------------------------------------
# 10. get_logs_by_trace_id convenience method
# ---------------------------------------------------------------------------

class TestGetLogsByTraceId:
    def test_returns_all_entries_for_trace(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        result = sample_log_provider.get_logs_by_trace_id("trace-009")
        assert result.total == 2  # rke-backend + inventory-svc share trace-009
        assert all(e.trace_id == "trace-009" for e in result.entries)

    def test_single_entry_trace(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        result = sample_log_provider.get_logs_by_trace_id("trace-001")
        assert result.total == 1

    def test_unknown_trace_returns_empty(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        result = sample_log_provider.get_logs_by_trace_id("trace-nonexistent")
        assert result.total == 0


# ---------------------------------------------------------------------------
# 11. get_log_by_id
# ---------------------------------------------------------------------------

class TestGetLogById:
    def test_returns_entry_for_valid_id(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        # Load one entry, then look it up by its stable ID
        all_entries = sample_log_provider.search_logs(LogSearchQuery()).entries
        assert all_entries, "Fixture must be non-empty"
        target = all_entries[0]
        found = sample_log_provider.get_log_by_id(target.id)
        assert found is not None
        assert found.id == target.id
        assert found.message == target.message

    def test_returns_none_for_unknown_id(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        result = sample_log_provider.get_log_by_id("0" * 64)
        assert result is None


# ---------------------------------------------------------------------------
# 12. Deterministic sort order
# ---------------------------------------------------------------------------

class TestSortOrder:
    def test_results_sorted_ascending_by_timestamp(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        result = sample_log_provider.search_logs(LogSearchQuery())
        timestamps = [e.timestamp for e in result.entries]
        assert timestamps == sorted(timestamps)

    def test_sort_is_stable_across_calls(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        r1 = sample_log_provider.search_logs(LogSearchQuery())
        r2 = sample_log_provider.search_logs(LogSearchQuery())
        assert [e.id for e in r1.entries] == [e.id for e in r2.entries]


# ---------------------------------------------------------------------------
# 13. Bad path raises FileNotFoundError
# ---------------------------------------------------------------------------

class TestBadPath:
    def test_nonexistent_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            LocalLogProvider(log_path="/tmp/does-not-exist-rca-agent-test.jsonl")


# ---------------------------------------------------------------------------
# 14. LogSearchQuery validation
# ---------------------------------------------------------------------------

class TestLogSearchQueryValidation:
    def test_inverted_time_range_raises(self) -> None:
        with pytest.raises(ValueError, match="start_time must be before end_time"):
            LogSearchQuery(
                start_time=_utc("2026-09-19T12:00:00Z"),
                end_time=_utc("2026-09-19T10:00:00Z"),
            )

    def test_equal_start_end_is_valid(self) -> None:
        t = _utc("2026-09-19T10:00:00Z")
        q = LogSearchQuery(start_time=t, end_time=t)
        assert q.start_time == q.end_time


# ---------------------------------------------------------------------------
# 15. Acceptance criteria (explicitly stated in the spec)
# ---------------------------------------------------------------------------

class TestAcceptanceCriteria:
    def test_search_payment_service_errors(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        """Acceptance: search_logs(service='payment-service', level='ERROR')
        must return only ERROR entries from payment-service."""
        result = sample_log_provider.search_logs(
            LogSearchQuery(service="payment-service", level="ERROR")
        )
        assert result.total > 0, "Expected at least one payment-service ERROR"
        for entry in result.entries:
            assert entry.service == "payment-service", f"Unexpected service: {entry.service}"
            assert entry.level == "ERROR", f"Unexpected level: {entry.level}"

    def test_result_total_matches_entries_length(
        self, sample_log_provider: LocalLogProvider
    ) -> None:
        result = sample_log_provider.search_logs(
            LogSearchQuery(service="payment-service", level="ERROR")
        )
        assert result.total == len(result.entries)

    def test_directory_provider_loads_all_files(
        self, fixtures_dir: Path
    ) -> None:
        """A provider pointed at a directory reads all *.jsonl files in it."""
        provider = LocalLogProvider(log_path=fixtures_dir)
        # Both sample_logs.jsonl and malformed_logs.jsonl live there.
        # valid lines = 20 (sample) + 3 (malformed) = 23
        result = provider.search_logs(LogSearchQuery())
        assert result.total == 23
