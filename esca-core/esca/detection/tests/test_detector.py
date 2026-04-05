from __future__ import annotations

from esca.detection.sce_detector import SCEDetector
from esca.detection.semantic_shift import BowEmbedder


def test_detector_requires_correct_rollout() -> None:
    detector = SCEDetector(tau=0.1, embedder=BowEmbedder(dim=512))
    text = "I will attempt. wait, let me reconsider. The right answer is 42."
    assert detector.detect(rollout_text=text, is_correct=False) is None


def test_detector_ignores_substrings() -> None:
    detector = SCEDetector(tau=0.0, embedder=BowEmbedder(dim=512), return_non_genuine=True)
    text = "The waiter brought water. Nothing to see here."
    # should not match 'wait' inside 'waiter'
    assert detector.detect(rollout_text=text, is_correct=True) is None


def test_detector_finds_genuine_sce() -> None:
    detector = SCEDetector(tau=0.2, embedder=BowEmbedder(dim=2048))
    text = (
        "We need to solve this. First, let's talk about oranges and bicycles and oceans. "
        "wait, I made an error. Let's compute: 6 * 7 = 42. Therefore the answer is 42."
    )

    event = detector.detect(rollout_text=text, is_correct=True, problem_id="p1", step=10)
    assert event is not None
    assert event.is_genuine
    assert event.problem_id == "p1"
    assert event.step == 10
    cm = event.correction_moment.lower()
    assert ("wait" in cm) or ("i made an error" in cm)
    assert event.semantic_shift > 0.0
