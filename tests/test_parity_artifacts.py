import json

import numpy as np
import pytest

from mlx_grounding_pi.parity import compare, numeric_difference


def test_difference_metrics():
    x = np.array([1.0, 2.0, 3.0])
    r = numeric_difference(x, x)
    assert r["max_abs_error"] == 0 and r["relative_l2_error"] == 0
    assert r["cosine_similarity"] == pytest.approx(1)
    with pytest.raises(ValueError, match="shape mismatch"):
        numeric_difference(x, np.ones((2,)))


def test_artifact_comparison_checks_inputs_and_does_not_invent_accuracy(tmp_path):
    a, b = tmp_path / "ref", tmp_path / "mlx"
    a.mkdir()
    b.mkdir()
    report = {
        "manifest_sha256": "same",
        "cases": [
            {
                "id": "case-one",
                "image_sha256": "photo",
                "instruction": "point",
                "result": {"raw_output": "text"},
            }
        ],
    }
    arrays = {
        "input_ids": np.array([[1, 2]]),
        "image_grid_thw": np.array([[1, 2, 2]]),
        "first_logits": np.arange(20, dtype=np.float32),
        "generated_ids": np.array([3, 4]),
    }
    for directory in [a, b]:
        (directory / "report.json").write_text(json.dumps(report))
        np.savez(directory / "case-one.npz", **arrays)
    comparison = compare(a, b, tmp_path / "comparison.json")
    case = comparison["cases"][0]
    assert (
        case["first_argmax_match"] and case["generated_ids_equal"] and case["top10_overlap"] == 10
    )
    assert "passed" not in case
    np.savez(b / "case-one.npz", **{**arrays, "input_ids": np.array([[2, 3]])})
    with pytest.raises(ValueError, match="model inputs differ"):
        compare(a, b, tmp_path / "wrong.json")
