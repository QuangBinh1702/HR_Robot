from src.human_detection.person_gate import PersonTemporalGate


def _person_det(height: int = 120, score: float = 0.8):
    return {
        "bbox": [10.0, 10.0, 80.0, 10.0 + float(height)],
        "score": score,
        "class_id": 0,
        "class_name": "person",
    }


def test_gate_requires_consecutive_hits():
    gate = PersonTemporalGate(
        min_bbox_height=80,
        required_consecutive_hits=2,
        miss_frames_to_reset=2,
    )

    _, s1 = gate.update([_person_det(120)])
    assert s1.trigger_active is False
    assert s1.consecutive_hits == 1

    _, s2 = gate.update([_person_det(120)])
    assert s2.trigger_active is True
    assert s2.consecutive_hits == 2


def test_gate_ignores_small_bbox_and_resets():
    gate = PersonTemporalGate(
        min_bbox_height=80,
        required_consecutive_hits=2,
        miss_frames_to_reset=2,
    )

    gate.update([_person_det(120)])
    gate.update([_person_det(120)])

    filtered1, s1 = gate.update([_person_det(60)])
    assert filtered1 == []
    assert s1.trigger_active is True
    assert s1.consecutive_misses == 1

    filtered2, s2 = gate.update([])
    assert filtered2 == []
    assert s2.trigger_active is False
    assert s2.consecutive_misses == 2
