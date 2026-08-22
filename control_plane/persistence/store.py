"""SQLite persistence for the control plane.

Saves projects, tasks (including OpenRouter usage records), and workflow state
to a local SQLite database so the dashboard survives restarts. This is a
plain-stdlib implementation (no ORM dependency).

The intent is a durable snapshot that is written on state changes and loaded on
startup, not a full transactional event store.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core.interfaces import (
    Project,
    Task,
    TaskStatus,
    UsageRecord,
)

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    goal TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    budget_usd REAL,
    task_ids TEXT NOT NULL DEFAULT '[]',
    created_at TEXT,
    updated_at TEXT,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    required_capability TEXT NOT NULL DEFAULT '',
    project_id TEXT,
    parent_task_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    priority INTEGER NOT NULL DEFAULT 0,
    payload TEXT NOT NULL DEFAULT '{}',
    result TEXT,
    error TEXT,
    dependencies TEXT NOT NULL DEFAULT '[]',
    usage TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT,
    started_at TEXT,
    completed_at TEXT
);
"""


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, default=str)


def _json_loads(text: Optional[str], default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


class PersistenceStore:
    """SQLite-backed store for control-plane state."""

    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # ----- Save -----

    def save_projects(self, projects: List[Project]) -> None:
        for p in projects:
            self._conn.execute(
                """
                INSERT INTO projects
                    (id, name, description, goal, status, budget_usd, task_ids,
                     created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name, description=excluded.description,
                    goal=excluded.goal, status=excluded.status,
                    budget_usd=excluded.budget_usd, task_ids=excluded.task_ids,
                    updated_at=excluded.updated_at, metadata=excluded.metadata
                """,
                (
                    p.id,
                    p.name,
                    p.description,
                    p.goal,
                    p.status.value if hasattr(p.status, "value") else str(p.status),
                    p.budget_usd,
                    _json_dumps(p.task_ids),
                    p.created_at.isoformat() if p.created_at else _now_iso(),
                    p.updated_at.isoformat() if p.updated_at else _now_iso(),
                    _json_dumps(p.metadata),
                ),
            )
        self._conn.commit()

    def save_tasks(self, tasks: List[Task]) -> None:
        for t in tasks:
            self._conn.execute(
                """
                INSERT INTO tasks
                    (id, name, description, required_capability, project_id,
                     parent_task_id, status, priority, payload, result, error,
                     dependencies, usage, metadata, created_at, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name, description=excluded.description,
                    required_capability=excluded.required_capability,
                    project_id=excluded.project_id, parent_task_id=excluded.parent_task_id,
                    status=excluded.status, priority=excluded.priority,
                    payload=excluded.payload, result=excluded.result, error=excluded.error,
                    dependencies=excluded.dependencies, usage=excluded.usage,
                    metadata=excluded.metadata, started_at=excluded.started_at,
                    completed_at=excluded.completed_at
                """,
                (
                    t.id,
                    t.name,
                    t.description,
                    t.required_capability,
                    t.project_id,
                    t.parent_task_id,
                    t.status.value if hasattr(t.status, "value") else str(t.status),
                    t.priority,
                    _json_dumps(t.payload),
                    _json_dumps(t.result),
                    t.error,
                    _json_dumps(t.dependencies),
                    _json_dumps(dataclass_to_dict(t.usage)) if t.usage else None,
                    _json_dumps(t.metadata),
                    t.created_at.isoformat() if t.created_at else _now_iso(),
                    t.started_at.isoformat() if t.started_at else None,
                    t.completed_at.isoformat() if t.completed_at else None,
                ),
            )
        self._conn.commit()

    # ----- Load -----

    def load_projects(self) -> List[Project]:
        rows = self._conn.execute(
            "SELECT id, name, description, goal, status, budget_usd, task_ids, "
            "created_at, updated_at, metadata FROM projects"
        ).fetchall()
        projects = []
        for r in rows:
            try:
                status = TaskStatus(r[4])
            except ValueError:
                status = TaskStatus.PENDING
            projects.append(
                Project(
                    id=r[0],
                    name=r[1] or "",
                    description=r[2] or "",
                    goal=r[3] or "",
                    status=status,
                    budget_usd=r[5],
                    task_ids=_json_loads(r[6], []),
                    created_at=_parse_dt(r[7]) or datetime.utcnow(),
                    updated_at=_parse_dt(r[8]) or datetime.utcnow(),
                    metadata=_json_loads(r[9], {}),
                )
            )
        return projects

    def load_tasks(self) -> List[Task]:
        rows = self._conn.execute(
            "SELECT id, name, description, required_capability, project_id, "
            "parent_task_id, status, priority, payload, result, error, "
            "dependencies, usage, metadata, created_at, started_at, completed_at "
            "FROM tasks"
        ).fetchall()
        tasks = []
        for r in rows:
            try:
                status = TaskStatus(r[6])
            except ValueError:
                status = TaskStatus.PENDING
            usage = None
            if r[12]:
                usage = dict_to_usage(_json_loads(r[12], {}))
            tasks.append(
                Task(
                    id=r[0],
                    name=r[1] or "",
                    description=r[2] or "",
                    required_capability=r[3] or "",
                    project_id=r[4],
                    parent_task_id=r[5],
                    status=status,
                    priority=r[7] or 0,
                    payload=_json_loads(r[8], {}),
                    result=_json_loads(r[9], None),
                    error=r[10],
                    dependencies=_json_loads(r[11], []),
                    usage=usage,
                    metadata=_json_loads(r[13], {}),
                    created_at=_parse_dt(r[14]) or datetime.utcnow(),
                    started_at=_parse_dt(r[15]),
                    completed_at=_parse_dt(r[16]),
                )
            )
        return tasks

    def count(self) -> Dict[str, int]:
        projects = self._conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
        tasks = self._conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        return {"projects": projects, "tasks": tasks}


def dataclass_to_dict(usage: UsageRecord) -> Dict[str, Any]:
    return {
        "model": usage.model,
        "provider": usage.provider,
        "session_id": usage.session_id,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "total_tokens": usage.total_tokens,
        "estimated_cost_usd": usage.estimated_cost_usd,
        "actual_cost_usd": usage.actual_cost_usd,
        "api_call_count": usage.api_call_count,
    }


def dict_to_usage(d: Dict[str, Any]) -> UsageRecord:
    return UsageRecord(
        model=d.get("model", ""),
        provider=d.get("provider", ""),
        session_id=d.get("session_id", ""),
        input_tokens=d.get("input_tokens", 0),
        output_tokens=d.get("output_tokens", 0),
        cache_read_tokens=d.get("cache_read_tokens", 0),
        cache_write_tokens=d.get("cache_write_tokens", 0),
        reasoning_tokens=d.get("reasoning_tokens", 0),
        total_tokens=d.get("total_tokens", 0),
        estimated_cost_usd=d.get("estimated_cost_usd", 0.0),
        actual_cost_usd=d.get("actual_cost_usd"),
        api_call_count=d.get("api_call_count", 0),
    )


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None
