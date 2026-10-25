"""Strict upstream GAM parsing and explicit original-image display coordinates."""

from ._vendor.protocol import FormatError, GroundingEntry, parse_answer

__all__ = ["FormatError", "GroundingEntry", "parse_answer", "to_pixels"]


def to_pixels(coordinates, width: int, height: int) -> tuple[float, ...]:
    """Match upstream visualizer: v/999*(size-1), not camera/robot coordinates.

    Coordinates are normalized to the original image by the released client.
    Do not silently infer a padded-image inverse transform or a 3D position here.
    """
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("image dimensions must be positive integers")
    if len(coordinates) not in (2, 4):
        raise ValueError("expected a point or box")
    if any(type(v) is not int or not 0 <= v <= 999 for v in coordinates):
        raise ValueError("coordinates must be integers in [0,999]")
    return tuple(
        v / 999 * (width - 1 if i % 2 == 0 else height - 1) for i, v in enumerate(coordinates)
    )
