"""Guard against SQLite batch migrations silently destroying index definitions.

Under SQLite, ``op.batch_alter_table()`` cannot always ``ALTER``. When an
operation in the block requires it, Alembic recreates the table: it reflects the
current schema, builds a new table from that reflection, copies the rows, and
swaps. PostgreSQL is unaffected — it issues a plain ``ALTER`` and never rebuilds.

SQLAlchemy's SQLite reflection cannot round-trip every index, so a rebuild
silently degrades the ones it cannot read back. Two classes are lost, and they
are not equally bad:

* **Expression indexes are dropped outright.** Reflection skips them with a
  warning ("Skipped unsupported reflection of expression-based index"), so the
  rebuilt table simply has no such index. ``ix_latency`` and
  ``ix_cumulative_llm_token_count_total`` on ``spans`` are in this class.
* **Sort order is flattened.** A ``DESC`` index reflects as ascending, so the
  rebuilt index exists but no longer matches what the ORM declares.

Both are silent and permanent: nothing errors, row data survives, and the
migration's own ``downgrade()`` re-runs a batch recreate without restoring
anything. A migrated database then disagrees with ``models.py`` forever, while
one built by ``create_all()`` — every test run, every fresh dev instance — keeps
the declaration. Two populations, different schemas.

Passing ``copy_from=`` makes Alembic rebuild from the table you hand it instead
of from reflection, preserving whatever that definition carries.

This module is the authoring-time guard. It is static analysis over the
migration sources, so it needs no database and cannot be fooled by the same
reflection blind spot it is checking for.

The check is revision-aware. A migration can only damage an index that exists
when it runs, so each batch operation is compared against the indexes created
*earlier* in the revision chain rather than against today's schema. Without that,
the two historical ``batch_alter_table("traces")`` calls in
``4ded9e43755f`` would be reported, when in fact they precede the creation of
that table's ordered index by eight revisions and cannot touch it.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Literal, Optional

import pytest
from sqlalchemy import Column, Index, MetaData
from sqlalchemy.sql.elements import TextClause, UnaryExpression

import phoenix.db.migrations
from phoenix.db.models import Base

_MIGRATIONS = Path(phoenix.db.migrations.__path__[0]) / "versions"

_LossKind = Literal["expression", "ordering"]

_LOSS_DESCRIPTION: dict[_LossKind, str] = {
    "expression": "expression index — reflection skips it, so a rebuild drops it entirely",
    "ordering": "declares sort order — reflection flattens it to ascending",
}

# An index key that is nothing but an identifier is a plain column reference,
# which SQLite reflection reads back correctly. Anything else is computed.
_BARE_COLUMN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TRAILING_DIRECTION = re.compile(r"^(?P<body>.*?)\s+(?P<direction>ASC|DESC)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# What reflection cannot round-trip
# ---------------------------------------------------------------------------


def _classify_expression(expr: object) -> Optional[_LossKind]:
    """How reflection would damage this one index key, if at all.

    ``text("start_time DESC")`` is a column reference wearing a direction, not an
    expression: SQLite reflects it back as a column and merely forgets the
    ``DESC``. Only a genuinely computed key — ``text("(end_time - start_time)")``
    — is skipped outright. Conflating the two would report the milder loss as the
    catastrophic one.
    """
    if isinstance(expr, Column):
        return None
    if isinstance(expr, UnaryExpression):
        modifier = getattr(expr, "modifier", None)
        if getattr(modifier, "__name__", "") in ("asc_op", "desc_op"):
            return "ordering" if isinstance(expr.element, Column) else "expression"
        return "expression"
    if isinstance(expr, TextClause):
        match = _TRAILING_DIRECTION.match(expr.text.strip())
        body = match.group("body").strip() if match else expr.text.strip()
        if not _BARE_COLUMN.match(body):
            return "expression"
        return "ordering" if match else None
    return "expression"


def _classify_index(index: Index) -> Optional[_LossKind]:
    """The worst loss the index would suffer, or ``None`` if it round-trips."""
    kinds = {kind for expr in index.expressions if (kind := _classify_expression(expr))}
    if "expression" in kinds:
        return "expression"
    return "ordering" if "ordering" in kinds else None


def _reflection_unsafe_indexes(metadata: MetaData) -> dict[str, dict[str, _LossKind]]:
    """Map each table to the indexes a reflection-based rebuild would damage.

    Derived from the live ORM metadata rather than a hand-maintained list, so a
    newly declared expression or ordered index extends this guard on its own.
    """
    unsafe: dict[str, dict[str, _LossKind]] = {}
    for table in metadata.sorted_tables:
        for index in table.indexes:
            if (kind := _classify_index(index)) and index.name:
                unsafe.setdefault(table.name, {})[index.name] = kind
    return unsafe


# ---------------------------------------------------------------------------
# The revision chain
# ---------------------------------------------------------------------------

_REVISION = re.compile(r'^revision:\s*str\s*=\s*"(?P<rev>[^"]+)"', re.MULTILINE)
_DOWN_REVISION = re.compile(r'^down_revision[^=\n]*=\s*(?:"(?P<rev>[^"]+)"|None)', re.MULTILINE)


@lru_cache(maxsize=1)
def _chain_position_by_path() -> dict[Path, int]:
    """Each migration module's index in the linear revision chain."""
    down_by_rev: dict[str, Optional[str]] = {}
    path_by_rev: dict[str, Path] = {}
    for path in sorted(_MIGRATIONS.glob("*.py")):
        source = path.read_text()
        if not (revision := _REVISION.search(source)):
            continue
        down = _DOWN_REVISION.search(source)
        down_by_rev[revision.group("rev")] = down.group("rev") if down else None
        path_by_rev[revision.group("rev")] = path

    children: dict[Optional[str], list[str]] = {}
    for rev, down_rev in down_by_rev.items():
        children.setdefault(down_rev, []).append(rev)

    positions: dict[Path, int] = {}
    current: Optional[str] = None
    while nxt := children.get(current):
        assert len(nxt) == 1, f"revision chain branches at {current!r}: {nxt}"
        current = nxt[0]
        positions[path_by_rev[current]] = len(positions)
    assert len(positions) == len(down_by_rev), "revision chain does not cover every migration"
    return positions


@lru_cache(maxsize=1)
def _index_creation_position() -> dict[str, int]:
    """Earliest chain position at which each index name appears in a migration.

    Name-based, which is what makes it robust to how the index is created —
    ``op.create_index``, raw ``CREATE INDEX`` DDL, or an inline ``sa.Index`` in a
    ``create_table`` are all found. An index absent from every migration is never
    present in a migrated database, so it cannot be damaged by one.
    """
    positions: dict[str, int] = {}
    names = {
        name for indexes in _reflection_unsafe_indexes(Base.metadata).values() for name in indexes
    }
    for path, position in _chain_position_by_path().items():
        source = path.read_text()
        for name in names:
            if name in source:
                positions[name] = min(positions.get(name, position), position)
    return positions


# ---------------------------------------------------------------------------
# What the migrations do
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BatchAlterCall:
    path: Path
    lineno: int
    table_name: Optional[str]
    has_copy_from: bool

    def __str__(self) -> str:
        return (
            f"{self.path.name}:{self.lineno} "
            f"batch_alter_table({self.table_name or '<non-literal>'!r})"
        )


def _batch_alter_calls(path: Path) -> Iterator[_BatchAlterCall]:
    """Every ``batch_alter_table`` call in one migration module."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != "batch_alter_table":
            continue
        table_name: Optional[str] = None
        if node.args and isinstance(node.args[0], ast.Constant):
            if isinstance(node.args[0].value, str):
                table_name = node.args[0].value
        for keyword in node.keywords:
            if (
                keyword.arg == "table_name"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ):
                table_name = keyword.value.value
        yield _BatchAlterCall(
            path=path,
            lineno=node.lineno,
            table_name=table_name,
            has_copy_from=any(k.arg == "copy_from" for k in node.keywords),
        )


def _all_batch_alter_calls() -> list[_BatchAlterCall]:
    return [call for path in sorted(_MIGRATIONS.glob("*.py")) for call in _batch_alter_calls(path)]


def _damageable_indexes(call: _BatchAlterCall) -> dict[str, _LossKind]:
    """Unsafe indexes on the call's table that already exist when it runs."""
    unsafe = _reflection_unsafe_indexes(Base.metadata).get(call.table_name or "", {})
    created = _index_creation_position()
    position = _chain_position_by_path().get(call.path)
    if position is None:
        return {}
    return {
        name: kind for name, kind in unsafe.items() if name in created and created[name] < position
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReflectionUnsafeIndexDetection:
    """The detector itself, checked against SQLite's actual behavior."""

    def test_both_loss_classes_are_represented_in_the_live_schema(self) -> None:
        unsafe = _reflection_unsafe_indexes(Base.metadata)
        found = {name: kind for indexes in unsafe.values() for name, kind in indexes.items()}
        # Sort order is the milder loss; a dropped expression index is the
        # catastrophic one. Both must be recognized, or the guard has quietly
        # narrowed to a single class.
        assert found.get("ix_traces_project_rowid_start_time") == "ordering"
        assert found.get("ix_project_sessions_project_id_end_time") == "ordering"
        assert found.get("ix_latency") == "expression"
        assert found.get("ix_cumulative_llm_token_count_total") == "expression"

    def test_desc_declared_via_the_orm_helper_is_detected(self) -> None:
        """``column.desc()`` builds a UnaryExpression, not a TextClause.

        Both spellings are in use, and a detector that only greps for
        ``text("... DESC")`` misses this one entirely.
        """
        unsafe = _reflection_unsafe_indexes(Base.metadata)
        assert (
            unsafe.get("experiment_logs", {}).get(
                "ix_experiment_logs_experiment_id_occurred_at_errors"
            )
            == "ordering"
        )

    def test_plain_column_indexes_are_not_flagged(self) -> None:
        unsafe = _reflection_unsafe_indexes(Base.metadata)
        flagged = {name for indexes in unsafe.values() for name in indexes}
        plain = {
            index.name
            for table in Base.metadata.sorted_tables
            for index in table.indexes
            if _classify_index(index) is None
        }
        assert plain, "expected the schema to contain ordinary indexes"
        assert not (plain & flagged)

    def test_a_column_named_description_is_not_mistaken_for_desc(self) -> None:
        assert _classify_index(Index("ix", TextClause("description"))) is None
        assert _classify_index(Index("ix2", TextClause("a, description DESC"))) == "expression"

    def test_computed_key_outranks_ordering(self) -> None:
        """A computed key is dropped whether or not it carries a direction."""
        assert _classify_index(Index("ix", TextClause("(a + b)"))) == "expression"
        assert _classify_index(Index("ix2", TextClause("(a + b) DESC"))) == "expression"
        assert _classify_index(Index("ix3", TextClause("a DESC"))) == "ordering"


class TestRevisionChain:
    def test_chain_is_linear_and_covers_every_migration(self) -> None:
        positions = _chain_position_by_path()
        assert len(positions) == len(list(_MIGRATIONS.glob("*.py")))
        assert sorted(positions.values()) == list(range(len(positions)))

    def test_every_unsafe_index_is_traceable_to_a_migration(self) -> None:
        """An index the migrations never mention would silently exempt its table."""
        unsafe = {
            name
            for indexes in _reflection_unsafe_indexes(Base.metadata).values()
            for name in indexes
        }
        missing = unsafe - set(_index_creation_position())
        assert not missing, (
            "these indexes are declared in models.py but created by no migration, so "
            f"migrated databases never have them: {sorted(missing)}"
        )


class TestBatchAlterTableSafety:
    def test_migrations_are_parseable_and_present(self) -> None:
        assert _all_batch_alter_calls(), f"no batch_alter_table calls found under {_MIGRATIONS}"

    def test_no_batch_alter_can_damage_an_existing_index(self) -> None:
        """The guard.

        Which operations trigger a rebuild is an Alembic implementation detail,
        not something a migration author can read off the call site — adding a
        column whose ``server_default`` wraps a ``ClauseElement`` is enough. So
        the requirement is placed on the table, not on the operation.
        """
        violations = {
            call: damageable
            for call in _all_batch_alter_calls()
            if not call.has_copy_from and (damageable := _damageable_indexes(call))
        }
        if violations:
            detail = "\n".join(
                f"  {call}\n"
                + "\n".join(
                    f"      would damage {name}: {_LOSS_DESCRIPTION[kind]}"
                    for name, kind in sorted(indexes.items())
                )
                for call, indexes in violations.items()
            )
            pytest.fail(
                "batch_alter_table on a table carrying an index that SQLite reflection "
                "cannot round-trip, without copy_from=:\n"
                f"{detail}\n\n"
                "Pass copy_from= with the table as it exists at this point in the "
                "migration history, so Alembic rebuilds from that definition rather "
                "than from reflection. Do not pass the current models.py __table__ "
                "unless the schema at this revision matches it — rebuilding from a "
                "future definition is worse than the drift being prevented."
            )

    def test_non_literal_table_names_are_visible_to_the_guard(self) -> None:
        """A computed table name would slip past the check above unnoticed."""
        unresolved = [call for call in _all_batch_alter_calls() if call.table_name is None]
        assert not unresolved, (
            "batch_alter_table called with a non-literal table name, which this "
            f"guard cannot resolve: {[str(c) for c in unresolved]}"
        )
