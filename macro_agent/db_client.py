"""SQLite Database Management Client for Macroeconomic Governance.

Operates on the central SQLite database (macro_governance.db) located in the
MetaTrader 5 Common Files folder, providing high-level Python utilities and CLI
commands for the AI agent to inspect, upsert, and prune calendar and news events.

Includes defensive transaction governance:
1. Automatic timestamped backup copy (.YYYYMMDD_HHMMSS.bkp) before any write operation.
2. Post-modification integrity check (PRAGMA integrity_check).
3. Immediate automatic rollback and restoration from the backup copy if any error or corruption occurs.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Dict, Generator, List, Optional
try:
    from zoneinfo import ZoneInfo
    MT5_TIMEZONE = ZoneInfo("Europe/Athens")
except ImportError:
    MT5_TIMEZONE = timezone(timedelta(hours=2))

_in_safe_transaction: bool = False


def get_default_db_path() -> Path:
    """Resolve the default MT5 Common Files database path."""
    env_common = os.getenv("MT5_COMMON_PATH")
    if env_common and Path(env_common).exists():
        base_dir = Path(env_common) / "Files"
    else:
        appdata = os.getenv("APPDATA", "")
        if appdata:
            base_dir = Path(appdata) / "MetaQuotes" / "Terminal" / "Common" / "Files"
        else:
            base_dir = Path.home() / ".mt5_common" / "Files"

    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "macro_governance.db"


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open a connection to the SQLite database and enable foreign keys / WAL mode."""
    target_path = db_path or get_default_db_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def verify_database_integrity(db_path: Optional[Path] = None) -> bool:
    """Run PRAGMA integrity_check on the database and return True if fully intact."""
    target_path = db_path or get_default_db_path()
    if not target_path.exists():
        return True

    with sqlite3.connect(str(target_path), timeout=10.0) as conn:
        conn.execute("PRAGMA wal_checkpoint(FULL);")
        cur = conn.execute("PRAGMA integrity_check;")
        row = cur.fetchone()
        return bool(row and row[0] == "ok")


@contextmanager
def safe_db_transaction(db_path: Optional[Path] = None) -> Generator[Path, None, None]:
    """Context manager creating a pre-modification backup (.YYYYMMDD_HHMMSS.bkp).

    Executes the database modification block, validates database integrity,
    and rolls back / restores the original file if any failure or corruption occurs.
    """
    global _in_safe_transaction
    target_path = db_path or get_default_db_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if _in_safe_transaction:
        yield target_path
        return

    _in_safe_transaction = True
    backup_file: Optional[Path] = None

    # 1. Create timestamped backup if database already exists and has content
    if target_path.exists() and target_path.stat().st_size > 0:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        backup_file = target_path.with_name(f"{target_path.name}.{ts}.bkp")
        try:
            with sqlite3.connect(str(target_path), timeout=5.0) as chk_conn:
                chk_conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception:
            pass
        shutil.copy2(target_path, backup_file)

    try:
        yield target_path

        # 2. Validate database integrity post-modification
        if target_path.exists():
            with sqlite3.connect(str(target_path), timeout=10.0) as test_conn:
                test_conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                cursor = test_conn.execute("PRAGMA integrity_check;")
                res = cursor.fetchone()
                if not res or res[0] != "ok":
                    raise sqlite3.DatabaseError(f"Database corruption detected by PRAGMA integrity_check: {res}")
    except Exception as err:
        # 3. Automatic Rollback: restore original file from backup
        if backup_file and backup_file.exists():
            try:
                wal_file = target_path.with_name(f"{target_path.name}-wal")
                shm_file = target_path.with_name(f"{target_path.name}-shm")
                for aux_file in (wal_file, shm_file):
                    if aux_file.exists():
                        try:
                            aux_file.unlink()
                        except Exception:
                            pass
                shutil.copy2(backup_file, target_path)
                print(f"[!] Operation failed: {err}. Successfully restored database from backup: {backup_file.name}")
            except Exception as restore_err:
                print(f"[CRITICAL] Failed to restore database from backup {backup_file.name}: {restore_err}")
        elif target_path.exists() and not backup_file:
            # If it was a newly created corrupted file, clean it up
            try:
                target_path.unlink()
            except Exception:
                pass
        raise
    finally:
        _in_safe_transaction = False


def init_schema(db_path: Optional[Path] = None) -> None:
    """Create the calendar_events and news_events tables if they do not exist."""
    with get_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS calendar_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                action TEXT NOT NULL DEFAULT 'BLOCK_ENTRIES',
                trailing_points INTEGER NOT NULL DEFAULT 0
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cal_lookup
            ON calendar_events (symbol, start_time, end_time);
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news_events (
                symbol TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                action TEXT NOT NULL DEFAULT 'BLOCK_ENTRIES',
                trailing_points INTEGER NOT NULL DEFAULT 0
            );
        """)
        # Backward-compatible migrations for existing databases
        try:
            conn.execute("ALTER TABLE calendar_events ADD COLUMN trailing_points INTEGER NOT NULL DEFAULT 0;")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE news_events ADD COLUMN trailing_points INTEGER NOT NULL DEFAULT 0;")
        except sqlite3.OperationalError:
            pass
        conn.commit()


def upsert_calendar_event(
    symbol: str,
    title: str,
    description: str,
    start_time: str,
    end_time: str,
    action: str = "BLOCK_ENTRIES",
    trailing_points: int = 0,
    event_id: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Insert or update a scheduled macroeconomic calendar event."""
    clean_sym = symbol.upper().strip()
    clean_action = action.upper().strip()

    valid_actions = {"BLOCK_ENTRIES", "TRAILING_STOP", "BREAKEVEN", "CLOSE_ALL", "ADVISORY_ONLY"}
    if clean_action not in valid_actions:
        raise ValueError(f"Invalid action '{clean_action}'. Must be one of {valid_actions}")

    with safe_db_transaction(db_path):
        init_schema(db_path)
        with get_connection(db_path) as conn:
            if event_id:
                conn.execute(
                    """
                    UPDATE calendar_events
                    SET symbol=?, title=?, description=?, start_time=?, end_time=?, action=?, trailing_points=?
                    WHERE id=?
                    """,
                    (clean_sym, title, description, start_time, end_time, clean_action, trailing_points, event_id),
                )
                conn.commit()
                return event_id
            else:
                cur = conn.execute(
                    """
                    INSERT INTO calendar_events
                    (symbol, title, description, start_time, end_time, action, trailing_points)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (clean_sym, title, description, start_time, end_time, clean_action, trailing_points),
                )
                conn.commit()
                return cur.lastrowid or 0


def delete_calendar_event(event_id: int, db_path: Optional[Path] = None) -> bool:
    """Delete a calendar event by ID."""
    with safe_db_transaction(db_path):
        with get_connection(db_path) as conn:
            cur = conn.execute("DELETE FROM calendar_events WHERE id = ?", (event_id,))
            conn.commit()
            return cur.rowcount > 0


def purge_expired_calendar_events(
    cutoff_time_str: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Delete all calendar events where end_time is in the past."""
    if not cutoff_time_str:
        cutoff_time_str = datetime.now(MT5_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")

    with safe_db_transaction(db_path):
        with get_connection(db_path) as conn:
            cur = conn.execute("DELETE FROM calendar_events WHERE end_time < ?", (cutoff_time_str,))
            conn.commit()
            return cur.rowcount


def get_calendar_events(
    symbol: Optional[str] = None,
    active_at_time: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Query scheduled calendar events with optional symbol and active-time filters."""
    init_schema(db_path)
    with get_connection(db_path) as conn:
        query = ("SELECT id, symbol, title, description, start_time, end_time, action, trailing_points "
                 "FROM calendar_events WHERE 1=1")
        params: List[Any] = []

        if symbol:
            query += " AND (symbol = ? OR symbol = 'GLOBAL')"
            params.append(symbol.upper().strip())

        if active_at_time:
            query += " AND ? >= start_time AND ? <= end_time"
            params.append(active_at_time)
            params.append(active_at_time)

        query += " ORDER BY start_time ASC"
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def upsert_news_event(
    symbol: str,
    title: str,
    description: str,
    action: str = "BLOCK_ENTRIES",
    trailing_points: int = 0,
    db_path: Optional[Path] = None,
) -> None:
    """Insert or replace an active breaking news blacklist record for a symbol."""
    clean_sym = symbol.upper().strip()
    clean_action = action.upper().strip()

    valid_actions = {"BLOCK_ENTRIES", "TRAILING_STOP", "BREAKEVEN", "CLOSE_ALL", "ADVISORY_ONLY"}
    if clean_action not in valid_actions:
        raise ValueError(f"Invalid action '{clean_action}'. Must be one of {valid_actions}")

    with safe_db_transaction(db_path):
        init_schema(db_path)
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO news_events (symbol, title, description, action, trailing_points)
                VALUES (?, ?, ?, ?, ?)
                """,
                (clean_sym, title, description, clean_action, trailing_points),
            )
            conn.commit()


def delete_news_event(symbol: str, db_path: Optional[Path] = None) -> bool:
    """Delete a breaking news record for a symbol (unblocking it)."""
    clean_sym = symbol.upper().strip()
    with safe_db_transaction(db_path):
        with get_connection(db_path) as conn:
            cur = conn.execute("DELETE FROM news_events WHERE symbol = ?", (clean_sym,))
            conn.commit()
            return cur.rowcount > 0


def get_news_events(
    symbol: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Query active breaking news records."""
    init_schema(db_path)
    with get_connection(db_path) as conn:
        query = "SELECT symbol, title, description, action, trailing_points FROM news_events WHERE 1=1"
        params: List[Any] = []

        if symbol:
            query += " AND (symbol = ? OR symbol = 'GLOBAL')"
            params.append(symbol.upper().strip())

        query += " ORDER BY symbol ASC"
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def list_backups(db_path: Optional[Path] = None) -> List[Path]:
    """List all available .bkp files for the target database, ordered chronologically."""
    target_path = db_path or get_default_db_path()
    parent_dir = target_path.parent
    if not parent_dir.exists():
        return []
    pattern = f"{target_path.name}.*.bkp"
    return sorted(parent_dir.glob(pattern), key=lambda p: p.name, reverse=True)


def restore_backup(backup_file: Path, db_path: Optional[Path] = None) -> bool:
    """Manually restore the database from a specified .bkp file."""
    target_path = db_path or get_default_db_path()
    if not backup_file.exists():
        raise FileNotFoundError(f"Backup file does not exist: {backup_file}")

    wal_file = target_path.with_name(f"{target_path.name}-wal")
    shm_file = target_path.with_name(f"{target_path.name}-shm")
    for aux_file in (wal_file, shm_file):
        if aux_file.exists():
            try:
                aux_file.unlink()
            except Exception:
                pass

    shutil.copy2(backup_file, target_path)
    return verify_database_integrity(target_path)


def print_database_status(db_path: Optional[Path] = None) -> None:
    """Print a clean CLI summary of the macro_governance.db database."""
    target_path = db_path or get_default_db_path()
    print("=" * 80)
    print(f"[*] Macro Governance SQLite Database: {target_path}")
    print(f"[*] Integrity Status: {'VALID (ok)' if verify_database_integrity(target_path) else 'CORRUPT / INVALID'}")
    print("=" * 80)

    if not target_path.exists():
        print("[!] Database file does not exist yet. Run 'init' to create it.")
        return

    cal_events = get_calendar_events(db_path=target_path)
    news_events = get_news_events(db_path=target_path)
    backups = list_backups(target_path)

    print(f"[+] Scheduled Calendar Events: {len(cal_events)}")
    for ev in cal_events:
        print(f"    - ID {ev['id']} [{ev['symbol']}] '{ev['title']}' ({ev['start_time']} -> {ev['end_time']}) "
              f"Action: {ev['action']} | Reason: {ev['description']}")

    print(f"\n[+] Active News Blacklist Events: {len(news_events)}")
    for nw in news_events:
        print(f"    - [{nw['symbol']}] '{nw['title']}' Action: {nw['action']} | Reason: {nw['description']}")

    print(f"\n[+] Available Backups ({len(backups)}):")
    for b in backups[:5]:
        print(f"    - {b.name} ({b.stat().st_size} bytes)")
    print("=" * 80)


def main() -> None:
    """CLI dispatcher for manual database management."""
    parser = argparse.ArgumentParser(description="Macro Governance SQLite Manager for LiveONNX-EA")
    parser.add_argument("--db", type=str, help="Custom path to macro_governance.db", default=None)
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("init", help="Initialize tables and indexes with safe backup")
    subparsers.add_parser("status", help="Print active calendar, news records, and backups")
    subparsers.add_parser("verify", help="Verify SQLite integrity via PRAGMA integrity_check")
    subparsers.add_parser("purge", help="Delete expired calendar events")
    subparsers.add_parser("list-backups", help="List all available timestamped .bkp files")

    restore_p = subparsers.add_parser("restore", help="Restore database from a specific .bkp file")
    restore_p.add_argument("--file", type=str, required=True, help="Path to .bkp file")

    add_cal = subparsers.add_parser("add-cal", help="Add or update scheduled calendar event")
    add_cal.add_argument("--symbol", required=True, help="Symbol code (e.g. EURUSD, GLOBAL)")
    add_cal.add_argument("--title", required=True, help="Event title")
    add_cal.add_argument("--desc", required=True, help="Event rationale description")
    add_cal.add_argument("--start", required=True, help="Start time (YYYY-MM-DD HH:MM:SS)")
    add_cal.add_argument("--end", required=True, help="End time (YYYY-MM-DD HH:MM:SS)")
    add_cal.add_argument("--action", default="BLOCK_ENTRIES", help="Protection action")
    add_cal.add_argument(
        "--trailing-points", type=int, default=0, help="Trailing distance in points for TRAILING_STOP"
    )
    add_cal.add_argument("--id", type=int, default=None, help="Optional event ID to update")

    del_cal = subparsers.add_parser("del-cal", help="Delete calendar event by ID")
    del_cal.add_argument("--id", type=int, required=True, help="Event ID to delete")

    add_news = subparsers.add_parser("add-news", help="Add or update breaking news blacklist event")
    add_news.add_argument("--symbol", required=True, help="Symbol code (e.g. EURUSD, GLOBAL)")
    add_news.add_argument("--title", required=True, help="News headline")
    add_news.add_argument("--desc", required=True, help="Detailed explanation of threat")
    add_news.add_argument("--action", default="BLOCK_ENTRIES", help="Protection action")
    add_news.add_argument(
        "--trailing-points", type=int, default=0, help="Trailing distance in points for TRAILING_STOP"
    )

    del_news = subparsers.add_parser("del-news", help="Remove symbol from news blacklist")
    del_news.add_argument("--symbol", required=True, help="Symbol code to unblock")

    args = parser.parse_args()
    db_path = Path(args.db) if args.db else None

    if args.command == "init":
        init_schema(db_path)
        print(f"[+] Initialized schema in {db_path or get_default_db_path()}")
    elif args.command == "status" or not args.command:
        print_database_status(db_path)
    elif args.command == "verify":
        is_ok = verify_database_integrity(db_path)
        print(f"[*] Database integrity check: {'OK' if is_ok else 'FAILED'}")
    elif args.command == "list-backups":
        b_list = list_backups(db_path)
        print(f"[+] Found {len(b_list)} backups:")
        for b in b_list:
            print(f"    - {b.name}")
    elif args.command == "restore":
        success = restore_backup(Path(args.file), db_path)
        print(f"[*] Restored from {args.file}: {'SUCCESS (ok)' if success else 'FAILED'}")
    elif args.command == "purge":
        count = purge_expired_calendar_events(db_path=db_path)
        print(f"[+] Purged {count} expired events.")
    elif args.command == "add-cal":
        ev_id = upsert_calendar_event(
            symbol=args.symbol,
            title=args.title,
            description=args.desc,
            start_time=args.start,
            end_time=args.end,
            action=args.action,
            trailing_points=args.trailing_points,
            event_id=args.id,
            db_path=db_path,
        )
        print(f"[+] Upserted calendar event ID {ev_id} for {args.symbol}")
    elif args.command == "del-cal":
        success = delete_calendar_event(args.id, db_path=db_path)
        print(f"[*] Deleted calendar event ID {args.id}: {success}")
    elif args.command == "add-news":
        upsert_news_event(
            symbol=args.symbol,
            title=args.title,
            description=args.desc,
            action=args.action,
            trailing_points=args.trailing_points,
            db_path=db_path,
        )
        print(f"[+] Added news blacklist for {args.symbol}: {args.title} (Action: {args.action})")
    elif args.command == "del-news":
        success = delete_news_event(args.symbol, db_path=db_path)
        print(f"[*] Removed news blacklist for {args.symbol}: {success}")


if __name__ == "__main__":
    main()
