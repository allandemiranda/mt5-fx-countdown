"""Unit tests for the isolated macro_agent sub-project and SQLite governance client."""

from pathlib import Path
import pytest

from macro_agent.db_client import (
    delete_calendar_event,
    delete_news_event,
    get_calendar_events,
    get_news_events,
    init_schema,
    purge_expired_calendar_events,
    upsert_calendar_event,
    upsert_news_event,
)
from macro_agent.fetcher import extract_currencies_from_symbol


def test_macro_db_init_and_tables(tmp_path: Path) -> None:
    """Verify that schema initialization creates required tables without error."""
    db_file = tmp_path / "test_macro.db"
    init_schema(db_file)
    assert db_file.exists()

    cal_events = get_calendar_events(db_path=db_file)
    news_events = get_news_events(db_path=db_file)
    assert cal_events == []
    assert news_events == []


def test_calendar_events_crud(tmp_path: Path) -> None:
    """Test inserting, updating, querying, and deleting scheduled calendar events."""
    db_file = tmp_path / "test_macro.db"

    ev_id = upsert_calendar_event(
        symbol="EURUSD",
        title="US Non-Farm Payrolls",
        description="High labor volatility",
        start_time="2026-09-04 12:00:00",
        end_time="2026-09-04 14:00:00",
        action="TRAILING_STOP",
        trailing_points=120,
        db_path=db_file,
    )
    assert ev_id > 0

    events = get_calendar_events(symbol="EURUSD", db_path=db_file)
    assert len(events) == 1
    assert events[0]["title"] == "US Non-Farm Payrolls"
    assert events[0]["action"] == "TRAILING_STOP"
    assert events[0]["trailing_points"] == 120

    # Update event action
    upsert_calendar_event(
        symbol="EURUSD",
        title="US Non-Farm Payrolls",
        description="High labor volatility updated",
        start_time="2026-09-04 12:00:00",
        end_time="2026-09-04 14:00:00",
        action="CLOSE_ALL",
        event_id=ev_id,
        db_path=db_file,
    )
    updated = get_calendar_events(symbol="EURUSD", db_path=db_file)
    assert len(updated) == 1
    assert updated[0]["action"] == "CLOSE_ALL"

    # Delete event
    deleted = delete_calendar_event(ev_id, db_path=db_file)
    assert deleted is True
    assert get_calendar_events(symbol="EURUSD", db_path=db_file) == []


def test_calendar_active_query_filter(tmp_path: Path) -> None:
    """Verify that querying with active_at_time matches exact time windows."""
    db_file = tmp_path / "test_macro.db"

    upsert_calendar_event(
        symbol="EURUSD",
        title="Active Event",
        description="Currently active",
        start_time="2026-09-04 12:00:00",
        end_time="2026-09-04 14:00:00",
        action="BLOCK_ENTRIES",
        db_path=db_file,
    )

    # Time before window
    assert get_calendar_events(symbol="EURUSD", active_at_time="2026-09-04 11:59:59", db_path=db_file) == []
    # Time inside window
    active = get_calendar_events(symbol="EURUSD", active_at_time="2026-09-04 12:30:00", db_path=db_file)
    assert len(active) == 1
    assert active[0]["title"] == "Active Event"
    # Time after window
    assert get_calendar_events(symbol="EURUSD", active_at_time="2026-09-04 14:00:01", db_path=db_file) == []


def test_news_events_crud(tmp_path: Path) -> None:
    """Test adding, querying, and removing breaking news blacklist records."""
    db_file = tmp_path / "test_macro.db"

    upsert_news_event(
        symbol="EURUSD",
        title="Geopolitical Escalation",
        description="Extreme erratic volatility",
        action="BLOCK_ENTRIES",
        db_path=db_file,
    )

    news = get_news_events(symbol="EURUSD", db_path=db_file)
    assert len(news) == 1
    assert news[0]["symbol"] == "EURUSD"
    assert news[0]["action"] == "BLOCK_ENTRIES"

    # Remove news
    deleted = delete_news_event("EURUSD", db_path=db_file)
    assert deleted is True
    assert get_news_events(symbol="EURUSD", db_path=db_file) == []


def test_purge_expired_calendar_events(tmp_path: Path) -> None:
    """Verify purging of past events."""
    db_file = tmp_path / "test_macro.db"

    upsert_calendar_event(
        symbol="EURUSD",
        title="Past Event",
        description="Expired",
        start_time="2026-01-01 10:00:00",
        end_time="2026-01-01 11:00:00",
        action="BLOCK_ENTRIES",
        db_path=db_file,
    )
    upsert_calendar_event(
        symbol="EURUSD",
        title="Future Event",
        description="Upcoming",
        start_time="2026-12-01 10:00:00",
        end_time="2026-12-01 11:00:00",
        action="BLOCK_ENTRIES",
        db_path=db_file,
    )

    purged = purge_expired_calendar_events(cutoff_time_str="2026-06-01 00:00:00", db_path=db_file)
    assert purged == 1

    remaining = get_calendar_events(db_path=db_file)
    assert len(remaining) == 1
    assert remaining[0]["title"] == "Future Event"


def test_invalid_action_raises_value_error(tmp_path: Path) -> None:
    """Verify that invalid actions raise ValueError."""
    db_file = tmp_path / "test_macro.db"

    with pytest.raises(ValueError, match="Invalid action"):
        upsert_calendar_event(
            symbol="EURUSD",
            title="Invalid",
            description="Bad Action",
            start_time="2026-09-04 12:00:00",
            end_time="2026-09-04 14:00:00",
            action="INVALID_ACTION",
            db_path=db_file,
        )


def test_macro_fetcher_extract_currencies() -> None:
    """Test extraction of currency components from symbols."""
    assert extract_currencies_from_symbol("EURUSD") == ["EUR", "USD"]
    assert extract_currencies_from_symbol("GBP/USD") == ["GBP", "USD"]
    assert extract_currencies_from_symbol("USDJPY.raw") == ["USD", "JPY"]
    assert extract_currencies_from_symbol("USD") == ["USD"]


def test_macro_agent_architectural_isolation() -> None:
    """Verify that macro_agent is cleanly isolated with its own README and runbooks."""
    root_dir = Path(__file__).resolve().parent.parent
    subproject_dir = root_dir / "macro_agent"

    assert (subproject_dir / "README.md").exists()
    assert (subproject_dir / "db_client.py").exists()
    assert (subproject_dir / "fetcher.py").exists()
    assert (subproject_dir / "prompts" / "UPDATE_ECONOMIC_CALENDAR.md").exists()
    assert (subproject_dir / "prompts" / "UPDATE_NEWS_GOVERNANCE.md").exists()

    # Verify that README contains the isolation notice
    readme_text = (subproject_dir / "README.md").read_text(encoding="utf-8")
    assert "ARCHITECTURAL BOUNDARY & ISOLATION NOTICE" in readme_text


def test_backup_created_on_modification(tmp_path: Path) -> None:
    """Verify that a timestamped .bkp copy is created before modifying an existing database."""
    from macro_agent.db_client import list_backups

    db_file = tmp_path / "test_macro.db"
    init_schema(db_file)

    # First modification
    upsert_calendar_event(
        symbol="EURUSD",
        title="Event 1",
        description="First",
        start_time="2026-09-04 12:00:00",
        end_time="2026-09-04 14:00:00",
        action="BLOCK_ENTRIES",
        db_path=db_file,
    )

    backups = list_backups(db_file)
    assert len(backups) >= 1
    assert backups[0].name.startswith("test_macro.db.")
    assert backups[0].name.endswith(".bkp")


def test_integrity_validation_and_rollback_on_failure(tmp_path: Path) -> None:
    """Verify that if an operation fails or corrupts the database, rollback restores the backup."""
    from macro_agent.db_client import safe_db_transaction, verify_database_integrity

    db_file = tmp_path / "test_macro.db"
    init_schema(db_file)

    upsert_calendar_event(
        symbol="EURUSD",
        title="Safe Event",
        description="Initial Good State",
        start_time="2026-09-04 12:00:00",
        end_time="2026-09-04 14:00:00",
        action="BLOCK_ENTRIES",
        db_path=db_file,
    )

    # State before failure
    initial_events = get_calendar_events(symbol="EURUSD", db_path=db_file)
    assert len(initial_events) == 1
    assert initial_events[0]["title"] == "Safe Event"

    # Simulate an operation that raises an exception or corrupts the file
    with pytest.raises(RuntimeError, match="Simulated crash"):
        with safe_db_transaction(db_file):
            # Corrupt file content or do bad operation
            with open(db_file, "wb") as f:
                f.write(b"CORRUPTED_GARBAGE_DATA_NOT_SQLITE")
            raise RuntimeError("Simulated crash")

    # Verify that database was restored from backup and is integral
    assert verify_database_integrity(db_file) is True
    restored_events = get_calendar_events(symbol="EURUSD", db_path=db_file)
    assert len(restored_events) == 1
    assert restored_events[0]["title"] == "Safe Event"


def test_manual_restore_backup(tmp_path: Path) -> None:
    """Verify manual restoration from a backup file."""
    from macro_agent.db_client import list_backups, restore_backup

    db_file = tmp_path / "test_macro.db"
    init_schema(db_file)

    # Insert Version 1
    upsert_calendar_event(
        symbol="EURUSD",
        title="Version 1",
        description="V1",
        start_time="2026-09-04 12:00:00",
        end_time="2026-09-04 14:00:00",
        action="BLOCK_ENTRIES",
        db_path=db_file,
    )
    v1_events = get_calendar_events(symbol="EURUSD", db_path=db_file)
    assert len(v1_events) == 1

    # Modify to V2 (which takes a backup of V1 state before modifying)
    upsert_calendar_event(
        symbol="EURUSD",
        title="Version 2",
        description="V2",
        start_time="2026-09-05 12:00:00",
        end_time="2026-09-05 14:00:00",
        action="BLOCK_ENTRIES",
        db_path=db_file,
    )
    v2_events = get_calendar_events(symbol="EURUSD", db_path=db_file)
    assert len(v2_events) == 2

    # The most recent backup was created immediately before inserting V2, so it contains V1
    backups = list_backups(db_file)
    assert len(backups) >= 1
    v1_backup = backups[0]

    # Manually restore to V1 backup
    success = restore_backup(v1_backup, db_path=db_file)
    assert success is True

    restored_events = get_calendar_events(symbol="EURUSD", db_path=db_file)
    assert len(restored_events) == 1
    assert restored_events[0]["title"] == "Version 1"
