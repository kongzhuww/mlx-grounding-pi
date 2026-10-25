"""Static checks only: importing these tests never connects to Modal."""

import ast
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_one_shot_modal_resource_and_upload_scope():
    tree = ast.parse((ROOT / "scripts/modal_reference.py").read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_reference"
    )
    decorator = function.decorator_list[0]
    settings = {
        item.arg: ast.literal_eval(item.value) for item in decorator.keywords if item.arg != "image"
    }
    assert settings["gpu"] == "A10G"
    assert settings["cpu"] == (4, 4) and settings["memory"] == (32768, 32768)
    assert settings["timeout"] == 1200 and settings["retries"] == 0
    assert settings["max_containers"] == 1 and settings["min_containers"] == 0
    assert settings["single_use_containers"] and not settings["include_source"]
    upload_loop = next(n for n in tree.body if isinstance(n, ast.For))
    assert ast.literal_eval(upload_loop.iter) == ["mlx_grounding_pi", "upstream", "examples"]
    text = (ROOT / "scripts/run_bounded_modal.py").read_text()
    assert "LIMIT_SECONDS = 1200" in text
    assert '"--yes"' in text and '"--detach"' not in text
    assert '"kongzhuww"' not in text


def test_cloud_watchdog_requires_explicit_paid_opt_in(monkeypatch):
    script = ROOT / "scripts/run_bounded_modal.py"
    monkeypatch.setattr(sys, "argv", [str(script)])

    def refuse_process(*args, **kwargs):
        pytest.fail("Missing approval must not spawn a process")

    monkeypatch.setattr("subprocess.Popen", refuse_process)
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(script), run_name="__main__")
    assert exc.value.code == 2
