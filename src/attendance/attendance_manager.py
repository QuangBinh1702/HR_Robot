"""
Attendance Manager - State machine for automated check-in/out & people counting.

Consumes fresh detect_faces() results, manages member attendance state,
and tracks headcount with overload alerts.

State machine per member:
    ABSENT → CANDIDATE (first recognition) → PRESENT (confirmed, log check-in)
    PRESENT → ABSENT (not seen for N minutes, log check-out)
"""

import threading
from datetime import datetime, timedelta
from typing import Optional

from config.settings import (
    ATTENDANCE_CONFIDENCE_THRESHOLD,
    MIN_ATTENDANCE_HITS,
    CHECKOUT_ABSENT_MINUTES,
    MAX_ROOM_CAPACITY,
    HEADCOUNT_PERSIST_INTERVAL,
    OVERLOAD_ALERT_COOLDOWN,
)
from src.database.models import session_scope
from src.database.repository import FaceRepository


# Member attendance states
STATE_ABSENT = "absent"
STATE_CANDIDATE = "candidate"
STATE_PRESENT = "present"


class AttendanceManager:
    """
    In-memory state machine for attendance tracking and people counting.

    Call process_results() ONLY on fresh inference frames (not cached frames).
    Uses edge-triggered overload alerts and throttled DB writes.
    """

    def __init__(self, repo: FaceRepository):
        self.repo = repo
        self.confidence_threshold = ATTENDANCE_CONFIDENCE_THRESHOLD
        self.min_hits = MIN_ATTENDANCE_HITS
        self.checkout_timeout = timedelta(minutes=CHECKOUT_ABSENT_MINUTES)
        self.max_capacity = MAX_ROOM_CAPACITY
        self.headcount_interval = timedelta(seconds=HEADCOUNT_PERSIST_INTERVAL)
        self.overload_cooldown = timedelta(seconds=OVERLOAD_ALERT_COOLDOWN)

        # Per-member state: {member_id: {...}}
        self.member_states: dict[int, dict] = {}

        # Global state
        self.last_headcount: Optional[int] = None
        self.last_headcount_write_at: Optional[datetime] = None
        self.last_overload_state: bool = False
        self.last_overload_alert_at: Optional[datetime] = None

        # Unknown person tracking (in-memory only)
        self.unknown_count: int = 0
        self.unknown_first_seen_at: Optional[datetime] = None
        self.unknown_last_seen_at: Optional[datetime] = None
        self.unknown_peak_count: int = 0

        # Thread safety for API access
        self._lock = threading.RLock()

        # Last processed summary (for API quick access)
        self.last_summary: Optional[dict] = None

        # Restore state from open attendance logs
        self._restore_from_db()

    def _restore_from_db(self):
        """
        Load open attendance logs on startup.
        - Logs from TODAY → restore as PRESENT (resume session)
        - Logs from PREVIOUS days → auto check-out (app was closed without checkout)
        """
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        with session_scope() as session:
            open_logs = self.repo.list_open_attendance_logs(session)
            stale_count = 0

            for log in open_logs:
                member = log.member
                member_name = member.full_name if member else f"ID:{log.member_id}"

                if log.check_in_time < today_start:
                    # Stale log from previous day → auto check-out at end of that day
                    checkout_time = log.check_in_time.replace(hour=23, minute=59, second=59)
                    self.repo.log_checkout(session, log.id, check_out_time=checkout_time)
                    stale_count += 1
                    print(f"[Attendance] Auto check-out (stale): {member_name} "
                          f"(checked in {log.check_in_time.strftime('%Y-%m-%d %H:%M')})")
                else:
                    # Today's log → restore as PRESENT
                    self.member_states[log.member_id] = {
                        "status": STATE_PRESENT,
                        "candidate_hits": self.min_hits,
                        "candidate_first_seen_at": log.check_in_time,
                        "last_seen_at": log.check_in_time,
                        "active_log_id": log.id,
                        "last_confidence": log.confidence_score,
                        "full_name": member_name,
                    }

        if stale_count > 0:
            print(f"[Attendance] Closed {stale_count} stale log(s) from previous day(s)")
        if self.member_states:
            names = [s["full_name"] for s in self.member_states.values()]
            print(f"[Attendance] Restored {len(names)} open session(s) today: {', '.join(names)}")

    def _get_or_init_state(self, member_id: int, full_name: str) -> dict:
        """Get existing member state or create ABSENT state."""
        if member_id not in self.member_states:
            self.member_states[member_id] = {
                "status": STATE_ABSENT,
                "candidate_hits": 0,
                "candidate_first_seen_at": None,
                "last_seen_at": None,
                "active_log_id": None,
                "last_confidence": None,
                "full_name": full_name,
            }
        return self.member_states[member_id]

    def process_results(self, results: list[dict], observed_at: datetime = None) -> dict:
        """
        Process one batch of fresh inference results.

        Args:
            results: list of dicts from pipeline.detect_faces()
                     Each has: name, confidence, member_id, bbox, ...
            observed_at: timestamp for this batch (defaults to utcnow)

        Returns:
            Summary dict with events:
            {
                "timestamp": datetime,
                "headcount": int,
                "known_count": int,
                "unknown_count": int,
                "is_overloaded": bool,
                "new_checkins": [...],
                "new_checkouts": [...],
                "overload_alert_triggered": bool,
            }
        """
        with self._lock:
            now = observed_at or datetime.utcnow()

            new_checkins = []
            new_checkouts = []

            # --- 1. Aggregate known faces (best confidence per member) ---
            known_faces: dict[int, dict] = {}  # member_id → best result
            unknown_results = []

            for r in results:
                member_id = r.get("member_id")
                if member_id is not None:
                    if member_id not in known_faces or r["confidence"] > known_faces[member_id]["confidence"]:
                        known_faces[member_id] = r
                else:
                    unknown_results.append(r)

            # --- 2. Update attendance for known faces ---
            seen_member_ids = set()

            for member_id, r in known_faces.items():
                confidence = r["confidence"]
                name = r["name"]
                seen_member_ids.add(member_id)

                # Skip if below attendance threshold
                if confidence < self.confidence_threshold:
                    continue

                state = self._get_or_init_state(member_id, name)
                state["last_seen_at"] = now
                state["last_confidence"] = confidence

                if state["status"] == STATE_ABSENT:
                    # Transition: ABSENT → CANDIDATE
                    state["status"] = STATE_CANDIDATE
                    state["candidate_hits"] = 1
                    state["candidate_first_seen_at"] = now

                elif state["status"] == STATE_CANDIDATE:
                    # Accumulate hits
                    state["candidate_hits"] += 1

                    if state["candidate_hits"] >= self.min_hits:
                        # Transition: CANDIDATE → PRESENT (check-in!)
                        state["status"] = STATE_PRESENT
                        checkin_event = self._do_checkin(member_id, confidence, now)
                        if checkin_event:
                            new_checkins.append(checkin_event)

                elif state["status"] == STATE_PRESENT:
                    # Already present, just refresh last_seen_at (done above)
                    pass

            # --- 3. Auto check-out for members not seen ---
            for member_id, state in list(self.member_states.items()):
                if member_id in seen_member_ids:
                    continue

                if state["status"] == STATE_CANDIDATE:
                    # Not confirmed yet, reset to ABSENT
                    state["status"] = STATE_ABSENT
                    state["candidate_hits"] = 0
                    state["candidate_first_seen_at"] = None

                elif state["status"] == STATE_PRESENT:
                    # Check timeout
                    if state["last_seen_at"] and (now - state["last_seen_at"]) >= self.checkout_timeout:
                        checkout_event = self._do_checkout(member_id, now)
                        if checkout_event:
                            new_checkouts.append(checkout_event)

            # --- 4. Track unknown faces (in-memory only) ---
            unk_count = len(unknown_results)
            self.unknown_count = unk_count
            if unk_count > 0:
                if self.unknown_first_seen_at is None:
                    self.unknown_first_seen_at = now
                self.unknown_last_seen_at = now
                self.unknown_peak_count = max(self.unknown_peak_count, unk_count)
            else:
                self.unknown_first_seen_at = None
                self.unknown_last_seen_at = None

            # --- 5. Headcount & overload ---
            headcount = len(results)
            is_overloaded = headcount > self.max_capacity
            overload_alert_triggered = False

            # Edge-triggered overload alert
            if is_overloaded and not self.last_overload_state:
                can_alert = (
                    self.last_overload_alert_at is None
                    or (now - self.last_overload_alert_at) >= self.overload_cooldown
                )
                if can_alert:
                    overload_alert_triggered = True
                    self.last_overload_alert_at = now

            self.last_overload_state = is_overloaded

            # Persist SpaceStatus (throttled)
            self._persist_headcount(headcount, is_overloaded, now)

            # --- 6. Build summary ---
            known_count = len(known_faces)

            summary = {
                "timestamp": now,
                "headcount": headcount,
                "known_count": known_count,
                "unknown_count": unk_count,
                "is_overloaded": is_overloaded,
                "new_checkins": new_checkins,
                "new_checkouts": new_checkouts,
                "overload_alert_triggered": overload_alert_triggered,
            }
            self.last_summary = summary
            return summary

    def _do_checkin(self, member_id: int, confidence: float, now: datetime) -> Optional[dict]:
        """Write check-in to DB and update state."""
        state = self.member_states[member_id]
        try:
            with session_scope() as session:
                log = self.repo.log_checkin(session, member_id, confidence, check_in_time=now)
                state["active_log_id"] = log.id
            return {
                "member_id": member_id,
                "full_name": state["full_name"],
                "confidence": confidence,
                "time": now,
            }
        except Exception as e:
            print(f"[Attendance] Check-in error for {state['full_name']}: {e}")
            return None

    def _do_checkout(self, member_id: int, now: datetime) -> Optional[dict]:
        """Write check-out to DB and reset state."""
        state = self.member_states[member_id]
        log_id = state.get("active_log_id")

        if log_id is None:
            # No active log, just reset
            state["status"] = STATE_ABSENT
            state["candidate_hits"] = 0
            state["active_log_id"] = None
            return None

        try:
            with session_scope() as session:
                log = self.repo.log_checkout(session, log_id, check_out_time=now)
                duration = log.duration_minutes if log else 0

            event = {
                "member_id": member_id,
                "full_name": state["full_name"],
                "duration_minutes": round(duration, 1) if duration else 0,
                "time": now,
            }

            # Reset state
            state["status"] = STATE_ABSENT
            state["candidate_hits"] = 0
            state["candidate_first_seen_at"] = None
            state["active_log_id"] = None

            return event
        except Exception as e:
            print(f"[Attendance] Check-out error for {state['full_name']}: {e}")
            return None

    def manual_checkin(self, member_id: int, confidence: float = 1.0, now: datetime = None) -> dict:
        """Force a manual check-in from touchscreen/admin UI."""
        with self._lock:
            now = now or datetime.utcnow()
            with session_scope() as session:
                member = self.repo.get_member_by_id(session, member_id)
                if member is None:
                    return {"success": False, "message": "Không tìm thấy thành viên"}

            state = self._get_or_init_state(member_id, member.full_name)
            if state["status"] == STATE_PRESENT:
                return {"success": False, "message": f"{member.full_name} đang ở trạng thái có mặt"}

            state["status"] = STATE_PRESENT
            state["candidate_hits"] = self.min_hits
            state["candidate_first_seen_at"] = now
            state["last_seen_at"] = now
            state["last_confidence"] = confidence

            event = self._do_checkin(member_id, confidence, now)
            if event is None:
                state["status"] = STATE_ABSENT
                state["candidate_hits"] = 0
                state["candidate_first_seen_at"] = None
                return {"success": False, "message": f"Check-in thủ công thất bại cho {member.full_name}"}

            return {
                "success": True,
                "message": f"Đã check-in thủ công cho {member.full_name}",
                "event": event,
            }

    def manual_checkout(self, member_id: int, now: datetime = None) -> dict:
        """Force a manual check-out from touchscreen/admin UI."""
        with self._lock:
            now = now or datetime.utcnow()
            state = self.member_states.get(member_id)
            if state is None or state["status"] != STATE_PRESENT:
                return {"success": False, "message": "Thành viên này hiện không ở trạng thái có mặt"}

            event = self._do_checkout(member_id, now)
            if event is None:
                return {"success": False, "message": f"Check-out thủ công thất bại cho {state['full_name']}"}

            return {
                "success": True,
                "message": f"Đã check-out thủ công cho {state['full_name']}",
                "event": event,
            }

    def _persist_headcount(self, headcount: int, is_overloaded: bool, now: datetime):
        """Write SpaceStatus to DB (throttled: on change or heartbeat interval)."""
        should_write = False

        if self.last_headcount is None:
            should_write = True
        elif headcount != self.last_headcount:
            should_write = True
        elif is_overloaded != self.last_overload_state:
            should_write = True
        elif self.last_headcount_write_at and (now - self.last_headcount_write_at) >= self.headcount_interval:
            should_write = True

        if not should_write:
            return

        try:
            with session_scope() as session:
                self.repo.update_headcount(session, headcount, self.max_capacity, timestamp=now)
            self.last_headcount = headcount
            self.last_headcount_write_at = now
        except Exception as e:
            print(f"[Attendance] Headcount write error: {e}")

    def get_status_summary(self) -> dict:
        """Get current attendance status (for API/dashboard)."""
        with self._lock:
            present_members = [
                {"member_id": mid, "full_name": s["full_name"], "since": s["candidate_first_seen_at"]}
                for mid, s in self.member_states.items()
                if s["status"] == STATE_PRESENT
            ]
            return {
                "present_count": len(present_members),
                "present_members": present_members,
                "unknown_count": self.unknown_count,
                "headcount": (self.last_headcount or 0),
                "is_overloaded": self.last_overload_state,
                "max_capacity": self.max_capacity,
            }
