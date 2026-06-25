import pytest

from gesim.checkpoints import resolve


def test_local_path_passthrough(tmp_path):
    f = tmp_path / "weights.safetensors"
    f.write_bytes(b"x")
    assert resolve(str(f)) == str(f)


def test_missing_local_path():
    with pytest.raises(FileNotFoundError):
        resolve("/nonexistent/weights.safetensors")


def test_empty_ref():
    with pytest.raises(ValueError):
        resolve("")


def test_hf_uri_parsing(monkeypatch):
    import gesim.checkpoints as ck

    calls = {}
    monkeypatch.setattr(
        ck,
        "hf_hub_download",
        lambda repo_id, filename: calls.setdefault("file", (repo_id, filename)) or "/tmp/f",
    )
    monkeypatch.setattr(
        ck, "snapshot_download", lambda repo_id: calls.setdefault("snap", repo_id) or "/tmp/d"
    )
    resolve("hf://org/repo/sub/file.pt")
    assert calls["file"] == ("org/repo", "sub/file.pt")
    resolve("hf://org/repo")
    assert calls["snap"] == "org/repo"


def test_hf_uri_invalid():
    with pytest.raises(ValueError):
        resolve("hf://justorg")
