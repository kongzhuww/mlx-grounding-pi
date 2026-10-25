import pytest

from scripts.first_run import verify_smoke, wait_for_weights


def test_smoke_requires_a_known_target_not_just_valid_grammar():
    truth = {"center_pixels": [120, 224], "tolerance_pixels": 30}
    result = {"valid": True, "predictions": [{"pixels": [[121, 225]]}]}
    assert verify_smoke(result, truth)["center_error_pixels"] < 2
    result["predictions"][0]["pixels"] = [[340, 224]]
    with pytest.raises(ValueError, match="exceeds"):
        verify_smoke(result, truth)
    with pytest.raises(ValueError, match="exactly one"):
        verify_smoke({"valid": True, "predictions": []}, truth)
    with pytest.raises(ValueError, match="GAM"):
        verify_smoke({"valid": False}, truth)


def test_wait_requires_complete_sizes_and_reports_failed_download(tmp_path, monkeypatch):
    header = {"shard.safetensors": {"file_bytes": 4}}
    (tmp_path / "shard.safetensors").write_bytes(b"1234")
    wait_for_weights(tmp_path, header, download_pid=123)
    (tmp_path / "shard.safetensors").write_bytes(b"123")

    def missing(pid, signal):
        raise ProcessLookupError

    monkeypatch.setattr("scripts.first_run.os.kill", missing)
    with pytest.raises(RuntimeError, match="download process exited"):
        wait_for_weights(tmp_path, header, download_pid=123)
