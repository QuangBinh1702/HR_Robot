"""
Tests for AttendanceManager state machine.
Uses in-memory SQLite — no camera or InsightFace needed.
"""

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from src.database.models import Base, Member, AttendanceLog, SpaceStatus
from src.database.repository import FaceRepository
from src.attendance.attendance_manager import (
    AttendanceManager, STATE_ABSENT, STATE_CANDIDATE, STATE_PRESENT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_session():
    """In-memory SQLite with all tables, returns (engine, Session)."""
    eng = create_engine("sqlite:///:memory:", echo=False)

    @event.listens_for(eng, "connect")
    def _pragma(conn, _):
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.close()

    Base.metadata.create_all(eng)
    Session = sessionmaker(bind=eng)
    return eng, Session


@pytest.fixture
def setup(db_session):
    """Create members and return (Session, repo, alice_id, bob_id)."""
    eng, Session = db_session
    s = Session()
    alice = Member(full_name="Alice", role="member")
    bob = Member(full_name="Bob", role="member")
    s.add_all([alice, bob])
    s.commit()
    alice_id, bob_id = alice.id, bob.id
    s.close()
    return Session, FaceRepository(), alice_id, bob_id


def _make_manager(Session, repo):
    """Create AttendanceManager with session_scope patched to use our in-memory DB."""
    _Session = Session

    from contextlib import contextmanager

    @contextmanager
    def _mock_session_scope():
        s = _Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    with patch("src.attendance.attendance_manager.session_scope", _mock_session_scope):
        mgr = AttendanceManager(repo)
    # Also patch for ongoing calls
    mgr._session_scope = _mock_session_scope
    # Override config to test defaults (env may differ)
    mgr.min_hits = 2
    mgr.confidence_threshold = 0.45
    mgr.checkout_timeout = timedelta(minutes=10)
    mgr.max_capacity = 20
    return mgr


def _process(mgr, results, observed_at=None):
    """Call process_results with session_scope patched."""
    with patch("src.attendance.attendance_manager.session_scope", mgr._session_scope):
        return mgr.process_results(results, observed_at=observed_at)


def _face(name, confidence, member_id):
    """Helper to create a fake detection result dict."""
    return {
        "name": name,
        "confidence": confidence,
        "member_id": member_id,
        "bbox": [10, 10, 100, 100],
        "score": 0.99,
        "embedding": None,
        "keypoints": None,
    }


# ---------------------------------------------------------------------------
# Tests: Check-in state machine
# ---------------------------------------------------------------------------

class TestCheckinFlow:
    """Verify ABSENT → CANDIDATE → PRESENT transitions."""

    def test_single_frame_stays_candidate(self, setup):
        Session, repo, alice_id, _ = setup
        mgr = _make_manager(Session, repo)

        t1 = datetime(2025, 6, 15, 9, 0, 0)
        summary = _process(mgr, [_face("Alice", 0.85, alice_id)], t1)

        assert summary["new_checkins"] == []
        assert mgr.member_states[alice_id]["status"] == STATE_CANDIDATE
        assert mgr.member_states[alice_id]["candidate_hits"] == 1

    def test_two_frames_triggers_checkin(self, setup):
        Session, repo, alice_id, _ = setup
        mgr = _make_manager(Session, repo)

        t1 = datetime(2025, 6, 15, 9, 0, 0)
        t2 = datetime(2025, 6, 15, 9, 0, 1)

        _process(mgr, [_face("Alice", 0.85, alice_id)], t1)
        summary = _process(mgr, [_face("Alice", 0.88, alice_id)], t2)

        assert len(summary["new_checkins"]) == 1
        assert summary["new_checkins"][0]["full_name"] == "Alice"
        assert mgr.member_states[alice_id]["status"] == STATE_PRESENT

    def test_no_duplicate_checkin_after_present(self, setup):
        Session, repo, alice_id, _ = setup
        mgr = _make_manager(Session, repo)

        t = datetime(2025, 6, 15, 9, 0, 0)
        for i in range(5):
            summary = _process(mgr, [_face("Alice", 0.85, alice_id)], t + timedelta(seconds=i))

        # Only the second frame should have triggered check-in
        total_checkins = sum(
            1 for i in range(5)
            for _ in [None]  # just count
        )
        assert mgr.member_states[alice_id]["status"] == STATE_PRESENT

        # Verify only 1 attendance log in DB
        s = Session()
        logs = s.query(AttendanceLog).filter(AttendanceLog.member_id == alice_id).all()
        s.close()
        assert len(logs) == 1

    def test_below_threshold_ignored(self, setup):
        Session, repo, alice_id, _ = setup
        mgr = _make_manager(Session, repo)
        mgr.confidence_threshold = 0.50

        t1 = datetime(2025, 6, 15, 9, 0, 0)
        t2 = datetime(2025, 6, 15, 9, 0, 1)

        _process(mgr, [_face("Alice", 0.30, alice_id)], t1)
        summary = _process(mgr, [_face("Alice", 0.30, alice_id)], t2)

        assert summary["new_checkins"] == []
        assert alice_id not in mgr.member_states or mgr.member_states[alice_id]["status"] == STATE_ABSENT


class TestCheckoutFlow:
    """Verify auto check-out after timeout."""

    def test_checkout_after_absent_timeout(self, setup):
        Session, repo, alice_id, _ = setup
        mgr = _make_manager(Session, repo)
        mgr.checkout_timeout = timedelta(minutes=10)

        t1 = datetime(2025, 6, 15, 9, 0, 0)
        t2 = datetime(2025, 6, 15, 9, 0, 1)
        t3 = datetime(2025, 6, 15, 9, 11, 0)  # 11 min later

        # Check in
        _process(mgr, [_face("Alice", 0.85, alice_id)], t1)
        _process(mgr, [_face("Alice", 0.85, alice_id)], t2)

        # Alice disappears, send empty results after timeout
        summary = _process(mgr, [], t3)

        assert len(summary["new_checkouts"]) == 1
        assert summary["new_checkouts"][0]["full_name"] == "Alice"
        assert summary["new_checkouts"][0]["duration_minutes"] > 0
        assert mgr.member_states[alice_id]["status"] == STATE_ABSENT

    def test_no_checkout_before_timeout(self, setup):
        Session, repo, alice_id, _ = setup
        mgr = _make_manager(Session, repo)
        mgr.checkout_timeout = timedelta(minutes=10)

        t1 = datetime(2025, 6, 15, 9, 0, 0)
        t2 = datetime(2025, 6, 15, 9, 0, 1)
        t3 = datetime(2025, 6, 15, 9, 5, 0)  # only 5 min later

        _process(mgr, [_face("Alice", 0.85, alice_id)], t1)
        _process(mgr, [_face("Alice", 0.85, alice_id)], t2)
        summary = _process(mgr, [], t3)

        assert summary["new_checkouts"] == []
        assert mgr.member_states[alice_id]["status"] == STATE_PRESENT

    def test_candidate_resets_when_disappears(self, setup):
        Session, repo, alice_id, _ = setup
        mgr = _make_manager(Session, repo)

        t1 = datetime(2025, 6, 15, 9, 0, 0)
        t2 = datetime(2025, 6, 15, 9, 0, 1)

        # One frame → candidate
        _process(mgr, [_face("Alice", 0.85, alice_id)], t1)
        assert mgr.member_states[alice_id]["status"] == STATE_CANDIDATE

        # Disappears → reset to absent
        _process(mgr, [], t2)
        assert mgr.member_states[alice_id]["status"] == STATE_ABSENT
        assert mgr.member_states[alice_id]["candidate_hits"] == 0


class TestMultipleFaces:
    """Verify multi-person and duplicate handling."""

    def test_best_confidence_per_member(self, setup):
        Session, repo, alice_id, _ = setup
        mgr = _make_manager(Session, repo)

        t1 = datetime(2025, 6, 15, 9, 0, 0)

        # Two detections of Alice in same frame, different confidence
        results = [
            _face("Alice", 0.70, alice_id),
            _face("Alice", 0.90, alice_id),
        ]
        _process(mgr, results, t1)

        state = mgr.member_states[alice_id]
        assert state["last_confidence"] == 0.90
        assert state["candidate_hits"] == 1  # counts as one hit

    def test_two_members_independent_checkin(self, setup):
        Session, repo, alice_id, bob_id = setup
        mgr = _make_manager(Session, repo)

        t1 = datetime(2025, 6, 15, 9, 0, 0)
        t2 = datetime(2025, 6, 15, 9, 0, 1)

        results = [_face("Alice", 0.85, alice_id), _face("Bob", 0.80, bob_id)]
        _process(mgr, results, t1)
        summary = _process(mgr, results, t2)

        names = {e["full_name"] for e in summary["new_checkins"]}
        assert names == {"Alice", "Bob"}


class TestUnknownTracking:
    """Verify unknown person in-memory tracking."""

    def test_unknown_counted(self, setup):
        Session, repo, _, _ = setup
        mgr = _make_manager(Session, repo)

        t1 = datetime(2025, 6, 15, 9, 0, 0)
        results = [_face("unknown", 0.30, None), _face("unknown", 0.25, None)]
        summary = _process(mgr, results, t1)

        assert summary["unknown_count"] == 2
        assert summary["known_count"] == 0
        assert mgr.unknown_peak_count == 2

    def test_unknown_no_attendance_log(self, setup):
        Session, repo, _, _ = setup
        mgr = _make_manager(Session, repo)

        t1 = datetime(2025, 6, 15, 9, 0, 0)
        t2 = datetime(2025, 6, 15, 9, 0, 1)
        results = [_face("unknown", 0.30, None)]
        _process(mgr, results, t1)
        _process(mgr, results, t2)

        s = Session()
        logs = s.query(AttendanceLog).all()
        s.close()
        assert len(logs) == 0


# ---------------------------------------------------------------------------
# Tests: Headcount & Overload
# ---------------------------------------------------------------------------

class TestHeadcount:
    """Verify headcount counting and DB persistence."""

    def test_headcount_includes_all_faces(self, setup):
        Session, repo, alice_id, _ = setup
        mgr = _make_manager(Session, repo)

        t1 = datetime(2025, 6, 15, 9, 0, 0)
        results = [_face("Alice", 0.85, alice_id), _face("unknown", 0.30, None)]
        summary = _process(mgr, results, t1)

        assert summary["headcount"] == 2

    def test_headcount_persisted_on_change(self, setup):
        Session, repo, alice_id, _ = setup
        mgr = _make_manager(Session, repo)

        t1 = datetime(2025, 6, 15, 9, 0, 0)
        t2 = datetime(2025, 6, 15, 9, 0, 1)

        _process(mgr, [_face("Alice", 0.85, alice_id)], t1)
        _process(mgr, [_face("Alice", 0.85, alice_id), _face("unknown", 0.3, None)], t2)

        s = Session()
        records = s.query(SpaceStatus).order_by(SpaceStatus.timestamp).all()
        s.close()

        assert len(records) == 2
        assert records[0].current_headcount == 1
        assert records[1].current_headcount == 2


class TestOverloadAlerts:
    """Verify edge-triggered overload alerts."""

    def test_overload_triggered_on_crossing(self, setup):
        Session, repo, alice_id, _ = setup
        mgr = _make_manager(Session, repo)
        mgr.max_capacity = 1

        t1 = datetime(2025, 6, 15, 9, 0, 0)
        results = [_face("Alice", 0.85, alice_id), _face("unknown", 0.3, None)]
        summary = _process(mgr, results, t1)

        assert summary["is_overloaded"] is True
        assert summary["overload_alert_triggered"] is True

    def test_no_repeated_alert_while_overloaded(self, setup):
        Session, repo, alice_id, _ = setup
        mgr = _make_manager(Session, repo)
        mgr.max_capacity = 1

        t1 = datetime(2025, 6, 15, 9, 0, 0)
        t2 = datetime(2025, 6, 15, 9, 0, 1)
        results = [_face("Alice", 0.85, alice_id), _face("unknown", 0.3, None)]

        s1 = _process(mgr, results, t1)
        s2 = _process(mgr, results, t2)

        assert s1["overload_alert_triggered"] is True
        assert s2["overload_alert_triggered"] is False  # no repeat

    def test_alert_re_triggers_after_recovery(self, setup):
        Session, repo, alice_id, _ = setup
        mgr = _make_manager(Session, repo)
        mgr.max_capacity = 1
        mgr.overload_cooldown = timedelta(seconds=0)  # no cooldown for test

        t1 = datetime(2025, 6, 15, 9, 0, 0)
        t2 = datetime(2025, 6, 15, 9, 0, 1)
        t3 = datetime(2025, 6, 15, 9, 0, 2)

        overloaded = [_face("Alice", 0.85, alice_id), _face("unknown", 0.3, None)]
        normal = [_face("Alice", 0.85, alice_id)]

        s1 = _process(mgr, overloaded, t1)   # cross threshold
        s2 = _process(mgr, normal, t2)        # recover
        s3 = _process(mgr, overloaded, t3)    # cross again

        assert s1["overload_alert_triggered"] is True
        assert s2["overload_alert_triggered"] is False
        assert s3["overload_alert_triggered"] is True


# ---------------------------------------------------------------------------
# Tests: Startup restore
# ---------------------------------------------------------------------------

class TestStartupRestore:
    """Verify AttendanceManager restores from open attendance logs."""

    def test_restores_today_log_as_present(self, setup):
        """Open log from TODAY should be restored as PRESENT."""
        Session, repo, alice_id, _ = setup

        # Insert an open log from today
        from datetime import datetime as dt
        now = dt.utcnow()
        today_checkin = now.replace(hour=8, minute=0, second=0, microsecond=0)

        s = Session()
        log = AttendanceLog(
            member_id=alice_id,
            check_in_time=today_checkin,
            confidence_score=0.90,
        )
        s.add(log)
        s.commit()
        s.close()

        mgr = _make_manager(Session, repo)

        assert alice_id in mgr.member_states
        assert mgr.member_states[alice_id]["status"] == STATE_PRESENT
        assert mgr.member_states[alice_id]["full_name"] == "Alice"

    def test_auto_checkout_stale_log_from_previous_day(self, setup):
        """Open log from YESTERDAY should be auto checked-out on startup."""
        Session, repo, alice_id, _ = setup

        # Insert an open log from yesterday
        yesterday = datetime.utcnow() - timedelta(days=1)
        yesterday_checkin = yesterday.replace(hour=9, minute=0, second=0, microsecond=0)

        s = Session()
        log = AttendanceLog(
            member_id=alice_id,
            check_in_time=yesterday_checkin,
            confidence_score=0.85,
        )
        s.add(log)
        s.commit()
        log_id = log.id
        s.close()

        mgr = _make_manager(Session, repo)

        # Alice should NOT be in member_states (stale log was closed)
        assert alice_id not in mgr.member_states

        # Verify log was checked out
        s = Session()
        closed_log = s.query(AttendanceLog).filter(AttendanceLog.id == log_id).first()
        s.close()
        assert closed_log.check_out_time is not None
        assert closed_log.duration_minutes is not None

    def test_no_duplicate_checkin_after_restore(self, setup):
        Session, repo, alice_id, _ = setup

        # Pre-existing open log from today
        from datetime import datetime as dt
        now = dt.utcnow()
        today_checkin = now.replace(hour=8, minute=0, second=0, microsecond=0)

        s = Session()
        log = AttendanceLog(
            member_id=alice_id,
            check_in_time=today_checkin,
            confidence_score=0.90,
        )
        s.add(log)
        s.commit()
        s.close()

        mgr = _make_manager(Session, repo)

        # Alice appears again — should NOT create new check-in
        t = now.replace(hour=9, minute=0, second=0, microsecond=0)
        summary = _process(mgr, [_face("Alice", 0.85, alice_id)], t)

        assert summary["new_checkins"] == []

        # Verify still only 1 log in DB
        s = Session()
        logs = s.query(AttendanceLog).filter(AttendanceLog.member_id == alice_id).all()
        s.close()
        assert len(logs) == 1


class TestGetStatusSummary:
    """Verify get_status_summary() output."""

    def test_summary_after_checkin(self, setup):
        Session, repo, alice_id, bob_id = setup
        mgr = _make_manager(Session, repo)

        t1 = datetime(2025, 6, 15, 9, 0, 0)
        t2 = datetime(2025, 6, 15, 9, 0, 1)
        results = [_face("Alice", 0.85, alice_id)]
        _process(mgr, results, t1)
        _process(mgr, results, t2)

        status = mgr.get_status_summary()
        assert status["present_count"] == 1
        assert status["present_members"][0]["full_name"] == "Alice"
        assert status["headcount"] == 1
