"""Tests for detectors/class_filter.py: effective_classes function."""

from __future__ import annotations

from rtsp_warden.detectors.class_filter import effective_classes


def test_both_none_returns_none() -> None:
    """Both camera_classes and detector_classes None -> no filter (return None)."""
    assert effective_classes(None, None) is None


def test_camera_none_returns_detector_classes() -> None:
    """camera_classes=None, detector_classes set -> use detector's list."""
    result = effective_classes(None, ["car", "truck"])
    assert result == ["car", "truck"]


def test_detector_none_returns_camera_classes() -> None:
    """camera_classes set, detector_classes=None -> use camera's list."""
    result = effective_classes(["person", "dog"], None)
    assert result == ["person", "dog"]


def test_intersection_both_set() -> None:
    """Both set -> intersection of classes."""
    result = effective_classes(["person", "dog"], ["dog", "car"])
    assert result == ["dog"]


def test_intersection_empty_means_match_nothing() -> None:
    """Empty intersection returns [] (match nothing)."""
    result = effective_classes(["person", "dog"], ["car", "truck"])
    assert result == []


def test_intersection_case_insensitive() -> None:
    """Intersection uses set comparison (case-sensitive matching)."""
    # effective_classes uses sorted(set & set), so case must match
    result = effective_classes(["Person"], ["person"])
    # "Person" != "person" in set comparison, so intersection is empty
    assert result == []


def test_intersection_preserves_sorted_order() -> None:
    """Result is sorted alphabetically."""
    result = effective_classes(["truck", "car", "person"], ["person", "car"])
    assert result == ["car", "person"]


def test_full_overlap() -> None:
    """Full overlap returns all classes (sorted)."""
    result = effective_classes(["car", "truck", "bus"], ["bus", "car", "truck"])
    assert result == ["bus", "car", "truck"]


def test_single_class_intersection() -> None:
    """Single overlapping class returns list with one element."""
    result = effective_classes(["person"], ["person", "car", "truck"])
    assert result == ["person"]


def test_camera_classes_with_extras() -> None:
    """Camera has classes not in detector -> only overlap returned."""
    result = effective_classes(["person", "dog", "cat"], ["person"])
    assert result == ["person"]
