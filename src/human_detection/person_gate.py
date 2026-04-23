"""Temporal gate for person detections."""

from dataclasses import dataclass


@dataclass
class PersonGateState:
    trigger_active: bool
    consecutive_hits: int
    consecutive_misses: int


class PersonTemporalGate:
    """Debounce person detections before triggering downstream face stages."""

    def __init__(
        self,
        min_bbox_height: int,
        required_consecutive_hits: int,
        miss_frames_to_reset: int,
    ):
        self.min_bbox_height = min_bbox_height
        self.required_consecutive_hits = max(1, required_consecutive_hits)
        self.miss_frames_to_reset = max(1, miss_frames_to_reset)

        self._consecutive_hits = 0
        self._consecutive_misses = 0
        self._trigger_active = False

    def update(self, detections: list[dict]) -> tuple[list[dict], PersonGateState]:
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            bbox_h = max(0.0, y2 - y1)
            if bbox_h >= self.min_bbox_height:
                filtered.append(det)

        has_person = len(filtered) > 0

        if has_person:
            self._consecutive_hits += 1
            self._consecutive_misses = 0
            if self._consecutive_hits >= self.required_consecutive_hits:
                self._trigger_active = True
        else:
            self._consecutive_misses += 1
            self._consecutive_hits = 0
            if self._consecutive_misses >= self.miss_frames_to_reset:
                self._trigger_active = False

        state = PersonGateState(
            trigger_active=self._trigger_active,
            consecutive_hits=self._consecutive_hits,
            consecutive_misses=self._consecutive_misses,
        )
        return filtered, state
