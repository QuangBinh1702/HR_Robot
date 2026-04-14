"""
Tests for FaceRepository attendance & space status methods.
"""

from datetime import datetime, timedelta

from src.database.models import AttendanceLog, SpaceStatus
from src.database.repository import FaceRepository


class TestLogCheckin:
    """Tests for log_checkin()."""

    def test_creates_open_log(self, session, repo, member_alice):
        log = repo.log_checkin(session, member_alice.id, confidence_score=0.85)

        assert log.id is not None
        assert log.member_id == member_alice.id
        assert log.confidence_score == 0.85
        assert log.check_in_time is not None
        assert log.check_out_time is None
        assert log.duration_minutes is None

    def test_custom_checkin_time(self, session, repo, member_alice):
        t = datetime(2025, 6, 15, 9, 0, 0)
        log = repo.log_checkin(session, member_alice.id, 0.90, check_in_time=t)

        assert log.check_in_time == t

    def test_idempotent_when_open_log_exists(self, session, repo, member_alice):
        log1 = repo.log_checkin(session, member_alice.id, 0.80)
        log2 = repo.log_checkin(session, member_alice.id, 0.95)

        assert log1.id == log2.id  # same record returned
        assert log1.confidence_score == 0.80  # original confidence kept

    def test_new_log_after_previous_closed(self, session, repo, member_alice):
        log1 = repo.log_checkin(session, member_alice.id, 0.80)
        repo.log_checkout(session, log1.id)

        log2 = repo.log_checkin(session, member_alice.id, 0.90)
        assert log2.id != log1.id

    def test_separate_logs_for_different_members(self, session, repo, member_alice, member_bob):
        log_a = repo.log_checkin(session, member_alice.id, 0.80)
        log_b = repo.log_checkin(session, member_bob.id, 0.75)

        assert log_a.id != log_b.id
        assert log_a.member_id == member_alice.id
        assert log_b.member_id == member_bob.id


class TestLogCheckout:
    """Tests for log_checkout()."""

    def test_closes_log_and_calculates_duration(self, session, repo, member_alice):
        t_in = datetime(2025, 6, 15, 9, 0, 0)
        t_out = datetime(2025, 6, 15, 9, 45, 0)
        log = repo.log_checkin(session, member_alice.id, 0.85, check_in_time=t_in)

        closed = repo.log_checkout(session, log.id, check_out_time=t_out)

        assert closed is not None
        assert closed.check_out_time == t_out
        assert abs(closed.duration_minutes - 45.0) < 0.01

    def test_returns_none_for_already_closed(self, session, repo, member_alice):
        log = repo.log_checkin(session, member_alice.id, 0.85)
        repo.log_checkout(session, log.id)

        result = repo.log_checkout(session, log.id)
        assert result is None

    def test_returns_none_for_missing_id(self, session, repo):
        result = repo.log_checkout(session, 99999)
        assert result is None

    def test_duration_fractional_minutes(self, session, repo, member_alice):
        t_in = datetime(2025, 6, 15, 9, 0, 0)
        t_out = datetime(2025, 6, 15, 9, 0, 30)  # 30 seconds
        log = repo.log_checkin(session, member_alice.id, 0.80, check_in_time=t_in)
        closed = repo.log_checkout(session, log.id, check_out_time=t_out)

        assert abs(closed.duration_minutes - 0.5) < 0.01


class TestOpenAttendanceLogs:
    """Tests for get_open_attendance_log() and list_open_attendance_logs()."""

    def test_get_open_returns_none_when_empty(self, session, repo, member_alice):
        result = repo.get_open_attendance_log(session, member_alice.id)
        assert result is None

    def test_get_open_returns_active_log(self, session, repo, member_alice):
        log = repo.log_checkin(session, member_alice.id, 0.85)
        found = repo.get_open_attendance_log(session, member_alice.id)

        assert found is not None
        assert found.id == log.id

    def test_get_open_returns_none_after_checkout(self, session, repo, member_alice):
        log = repo.log_checkin(session, member_alice.id, 0.85)
        repo.log_checkout(session, log.id)

        found = repo.get_open_attendance_log(session, member_alice.id)
        assert found is None

    def test_list_open_returns_only_open(self, session, repo, member_alice, member_bob):
        log_a = repo.log_checkin(session, member_alice.id, 0.80)
        log_b = repo.log_checkin(session, member_bob.id, 0.75)
        repo.log_checkout(session, log_a.id)  # close Alice

        open_logs = repo.list_open_attendance_logs(session)
        open_ids = [l.id for l in open_logs]

        assert log_b.id in open_ids
        assert log_a.id not in open_ids

    def test_list_open_empty_when_all_closed(self, session, repo, member_alice):
        log = repo.log_checkin(session, member_alice.id, 0.85)
        repo.log_checkout(session, log.id)

        assert repo.list_open_attendance_logs(session) == []


class TestUpdateHeadcount:
    """Tests for update_headcount() and get_latest_space_status()."""

    def test_inserts_new_record(self, session, repo):
        status = repo.update_headcount(session, current_headcount=5, max_capacity=20)

        assert status.id is not None
        assert status.current_headcount == 5
        assert status.max_capacity == 20
        assert status.is_overloaded is False

    def test_overloaded_flag(self, session, repo):
        status = repo.update_headcount(session, current_headcount=25, max_capacity=20)
        assert status.is_overloaded is True

    def test_exactly_at_capacity_not_overloaded(self, session, repo):
        status = repo.update_headcount(session, current_headcount=20, max_capacity=20)
        assert status.is_overloaded is False

    def test_append_only_multiple_records(self, session, repo):
        repo.update_headcount(session, 3, 20)
        repo.update_headcount(session, 5, 20)
        repo.update_headcount(session, 2, 20)

        count = session.query(SpaceStatus).count()
        assert count == 3

    def test_get_latest_returns_newest(self, session, repo):
        t1 = datetime(2025, 6, 15, 9, 0, 0)
        t2 = datetime(2025, 6, 15, 9, 1, 0)
        repo.update_headcount(session, 3, 20, timestamp=t1)
        repo.update_headcount(session, 7, 20, timestamp=t2)

        latest = repo.get_latest_space_status(session)
        assert latest.current_headcount == 7
        assert latest.timestamp == t2

    def test_get_latest_returns_none_when_empty(self, session, repo):
        assert repo.get_latest_space_status(session) is None

    def test_custom_timestamp(self, session, repo):
        t = datetime(2025, 1, 1, 12, 0, 0)
        status = repo.update_headcount(session, 10, 20, timestamp=t)
        assert status.timestamp == t
