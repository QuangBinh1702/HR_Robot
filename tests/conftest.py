"""
Shared test fixtures for HR Robot tests.
Uses in-memory SQLite for isolation.
"""

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.models import Base, Member, MemberEmbedding, AttendanceLog, SpaceStatus
from src.database.repository import FaceRepository


@pytest.fixture
def engine():
    """In-memory SQLite engine with WAL-like pragmas."""
    eng = create_engine("sqlite:///:memory:", echo=False)

    @event.listens_for(eng, "connect")
    def _set_pragma(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.close()

    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def session(engine):
    """A fresh DB session that rolls back after each test."""
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.rollback()
    s.close()


@pytest.fixture
def repo():
    """FaceRepository instance."""
    return FaceRepository()


@pytest.fixture
def member_alice(session):
    """Pre-created member 'Alice'."""
    m = Member(full_name="Alice", role="member")
    session.add(m)
    session.flush()
    return m


@pytest.fixture
def member_bob(session):
    """Pre-created member 'Bob'."""
    m = Member(full_name="Bob", role="member")
    session.add(m)
    session.flush()
    return m
