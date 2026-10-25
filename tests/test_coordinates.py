import pytest

from mlx_grounding_pi.coordinates import FormatError, parse_answer, to_pixels


def answer(payload):
    return f"<|object_ref_start|>red block<|object_ref_end|><|box_start|>{payload}<|box_end|>"


def test_point_box_multiple_and_none():
    assert parse_answer(answer("<300><425>"), "point")[0].coordinates == ((300, 425),)
    assert parse_answer(answer("<1><2><500><600>"), "bbox")[0].coordinates == ((1, 2, 500, 600),)
    assert parse_answer(answer("<10><20>,<30><40>"), "point")[0].coordinates == ((10, 20), (30, 40))
    assert parse_answer(answer("None"), "point")[0].coordinates is None


@pytest.mark.parametrize(
    "text,task",
    [
        (answer("<1000><20>"), "point"),
        (answer("<01><20>"), "point"),
        (answer("<1>"), "point"),
        (answer("<1><2>") + "trailing", "point"),
        (answer("<5><2><1><10>"), "bbox"),
        (answer("<1><2>")[:-2], "point"),
    ],
)
def test_invalid_output_is_not_a_target(text, task):
    with pytest.raises(FormatError):
        parse_answer(text, task)


def test_pixels_use_upstream_display_convention():
    assert to_pixels((0, 999), 640, 480) == (0, 479)
    assert to_pixels((999, 999), 1, 1) == (0, 0)
    with pytest.raises(ValueError):
        to_pixels((1000, 0), 640, 480)
