import pytest

from mlx_grounding_pi.evaluation import assess, box_iou, instruction_for


def result(*pixels):
    return {"valid": True, "predictions": [{"pixels": list(pixels), "coordinates": [[1, 2]]}]}


def test_iou_and_order_independent_box_matching():
    a, b = [0, 0, 10, 10], [20, 20, 30, 30]
    assert box_iou(a, a) == 1 and box_iou(a, b) == 0
    assessed = assess(result(b, a), {"boxes": [a, b], "min_iou": 0.7})
    assert assessed["passed"] and assessed["min_matched_iou"] == 1
    assert not assess(result(a), {"boxes": [a, b], "min_iou": 0.7})["passed"]


def test_point_error_and_negative_targets():
    assert assess(result([10, 21]), {"points": [[10, 20]], "max_distance_pixels": 2})["passed"]
    assert not assess(result([10, 30]), {"points": [[10, 20]], "max_distance_pixels": 2})["passed"]
    absent = {"valid": True, "predictions": [{"pixels": None, "coordinates": None}]}
    assert assess(absent, {"absent": True})["passed"]
    assert not assess(result([1, 2]), {"absent": True})["passed"]


def test_photo_review_is_not_automatic_accuracy():
    assert assess(result([10, 20]), None) == {"status": "requires_visual_review", "passed": None}
    assert assess({"valid": False}, None)["status"] == "invalid_output"


def test_instruction_contract():
    assert instruction_for({"task": "point", "phrase": "red square"}).startswith(
        "Point to the target"
    )
    assert instruction_for({"task": "bbox", "prompt": "custom"}) == "custom"
    with pytest.raises(ValueError):
        instruction_for({"task": "bbox", "phrase": "x", "prompt": "y"})
