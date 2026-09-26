"""``models`` pull/verify against a local HTTP server (resume, sha256, zips, stamps)."""

from __future__ import annotations

import io
import os
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from ops_testkit import FileServer, sha, write_manifest

from aivtube.ops import models as M
from aivtube.ops.models import ModelError, ModelManifest, cuda_variant

REPO = Path(__file__).resolve().parents[3]
BLOB = os.urandom(300_000)


@pytest.fixture
def server() -> Iterator[FileServer]:
    srv = FileServer({"/blob.bin": BLOB})
    yield srv
    srv.close()


def manifest(root: Path, url: str, **over: object) -> ModelManifest:
    fields: dict[str, object] = {
        "url": url,
        "sha256": sha(BLOB),
        "size": len(BLOB),
        "licence": "MIT",
        "required_by": ["vad.model"],
        "dest": "models/vad/blob.bin",
        **over,
    }
    return ModelManifest.load(write_manifest(root, {"blob": fields}), root=root)


def test_real_manifest_loads_and_selects_variants() -> None:
    m = ModelManifest.load(REPO / "models" / "manifest.toml", root=REPO)
    assert {"silero_vad", "llm_4b", "llm_30b", "llama_cpp_cuda13"} <= set(m.names())
    new = m.for_variant("cuda-13.4")
    old = m.for_variant("cuda-12.4")
    assert "llama_cpp_cuda13" in new and "cudart_cuda13" in new and "llama_cpp_cuda12" not in new
    assert "llama_cpp_cuda12" in old and "cudart_cuda12" in old and "cudart_cuda13" not in old
    order = [n for n in new if n in ("silero_vad", "llm_4b", "llm_30b")]
    assert order == ["silero_vad", "llm_4b", "llm_30b"]  # the 4B before the 30B
    assert [e.name for e in m.required_by("llm.servers.local30b")] == ["llm_30b"]
    with_exe = {e.name for e in m.required_by("llm.servers.local30b", prefix=True)}
    assert {"llm_30b", "llama_cpp_cuda13", "cudart_cuda12"} <= with_exe
    assert not m.entry("typhoon_rt_encoder").pinned


@pytest.mark.parametrize("driver, variant", [(581, "cuda-13.4"), (580, "cuda-13.4"),
                                             (579, "cuda-12.4"), (None, "cuda-12.4")])
def test_cuda_variant_by_driver(driver: int | None, variant: str) -> None:
    assert cuda_variant(driver) == variant


def test_pull_verifies_and_skips_when_done(tmp_path: Path, server: FileServer) -> None:
    m = manifest(tmp_path, server.url + "/blob.bin")
    seen: list[tuple[str, int, int]] = []
    m.pull(["blob"], progress=lambda n, d, t: seen.append((n, d, t)))
    dest = tmp_path / "models" / "vad" / "blob.bin"
    assert dest.read_bytes() == BLOB
    assert seen[-1] == ("blob", len(BLOB), len(BLOB))
    assert m.verify(["blob"]) == {"blob": True}
    n = len(server.requests)
    m.pull(["blob"])
    assert len(server.requests) == n  # verified: no request
    assert (tmp_path / "models" / ".verified.json").is_file()


def test_resume_from_partial_file(tmp_path: Path, server: FileServer) -> None:
    m = manifest(tmp_path, server.url + "/blob.bin")
    part = tmp_path / "models" / "vad" / "blob.bin.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(BLOB[:120_000])
    assert m.check("blob").status == "partial"
    m.pull(["blob"])
    assert server.requests == [("/blob.bin", "bytes=120000-", 206)]
    assert (tmp_path / "models" / "vad" / "blob.bin").read_bytes() == BLOB
    assert m.check("blob").status == "ok"


def test_resume_after_a_dropped_connection(tmp_path: Path) -> None:
    srv = FileServer({"/blob.bin": BLOB}, cut_after=100_000, cuts=2)
    try:
        m = manifest(tmp_path, srv.url + "/blob.bin")
        m.pull(["blob"], retries=5)
        codes = [(r, c) for _, r, c in srv.requests]
        assert codes[0] == (None, 200)
        assert codes[1] == ("bytes=100000-", 206)
        assert codes[2] == ("bytes=200000-", 206)
        assert (tmp_path / "models" / "vad" / "blob.bin").read_bytes() == BLOB
    finally:
        srv.close()


def test_server_ignoring_range_restarts_the_file(tmp_path: Path) -> None:
    srv = FileServer({"/blob.bin": BLOB}, ignore_range=True)
    try:
        m = manifest(tmp_path, srv.url + "/blob.bin")
        part = tmp_path / "models" / "vad" / "blob.bin.part"
        part.parent.mkdir(parents=True)
        part.write_bytes(b"x" * 1000)
        m.pull(["blob"])
        assert (tmp_path / "models" / "vad" / "blob.bin").read_bytes() == BLOB
    finally:
        srv.close()


def test_a_server_that_keeps_sending_too_little_is_given_up_on(tmp_path: Path) -> None:
    """Short 200 answers without an error must count as retries (no endless loop)."""
    srv = FileServer({"/blob.bin": BLOB}, ignore_range=True)
    try:
        m = manifest(tmp_path, srv.url + "/blob.bin", size=len(BLOB) + 10)
        with pytest.raises(ModelError, match="incomplete"):
            m.pull(["blob"], retries=1)
        assert len(srv.requests) == 2
    finally:
        srv.close()


def test_a_second_puller_is_refused_while_one_downloads(tmp_path: Path, server: FileServer) -> None:
    m = manifest(tmp_path, server.url + "/blob.bin")
    other = M._FileLock(m.lock_path(m.entry("blob")))  # another process's pull
    assert other.acquire()
    try:
        assert m.downloading("blob") is True
        with pytest.raises(M.ModelBusy, match="already being downloaded"):
            m.pull_one("blob")
        assert server.requests == []
    finally:
        other.release()
    assert m.downloading("blob") is False
    assert m.pull_one("blob").status == "ok"


def test_sha_mismatch_fails_and_removes_the_file(tmp_path: Path, server: FileServer) -> None:
    m = manifest(tmp_path, server.url + "/blob.bin", sha256="0" * 64)
    with pytest.raises(ModelError, match="sha"):
        m.pull(["blob"])
    assert not (tmp_path / "models" / "vad" / "blob.bin").exists()


def test_corrupt_existing_file_is_downloaded_again(tmp_path: Path, server: FileServer) -> None:
    m = manifest(tmp_path, server.url + "/blob.bin")
    dest = tmp_path / "models" / "vad" / "blob.bin"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"y" * len(BLOB))
    assert m.check("blob").status == "sha"
    m.pull(["blob"])
    assert dest.read_bytes() == BLOB


def test_404_fails_without_retrying(tmp_path: Path, server: FileServer) -> None:
    m = manifest(tmp_path, server.url + "/missing.bin")
    with pytest.raises(ModelError, match="404"):
        m.pull(["blob"], retries=3)
    assert len(server.requests) == 1


def test_unpinned_entry_reports_its_digest(tmp_path: Path, server: FileServer) -> None:
    m = manifest(tmp_path, server.url + "/blob.bin", sha256="", size=0)
    result = m.pull_one("blob")
    assert result.status == "unpinned" and result.sha256 == sha(BLOB)
    assert m.verify(["blob"]) == {"blob": True}


def test_quick_check_hashes_small_files_and_trusts_stamps(
    tmp_path: Path, server: FileServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = manifest(tmp_path, server.url + "/blob.bin")
    m.pull(["blob"])
    (tmp_path / "models" / ".verified.json").unlink()
    monkeypatch.setattr(M, "CHEAP_SHA_BYTES", 1000)  # treat the blob as a "big" file
    quick = m.check("blob", full=False)
    assert quick.status == "ok" and "not checked" in quick.detail
    assert m.check("blob", full=True).detail == "verified"  # writes the stamp
    assert m.check("blob", full=False).detail == "verified"  # from the stamp
    dest = tmp_path / "models" / "vad" / "blob.bin"
    dest.write_bytes(b"z" * len(BLOB))  # same size, new content and mtime
    os.utime(dest, ns=(1, 1))
    quick = m.check("blob", full=False)
    assert quick.status == "ok" and "not checked" in quick.detail  # the stamp is stale
    assert m.check("blob", full=True).status == "sha"


def _zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_zip_entries_are_extracted_once(tmp_path: Path) -> None:
    data = _zip({"llama-server.exe": b"MZ fake", "ggml-cuda.dll": b"dll"})
    srv = FileServer({"/llama.zip": data})
    try:
        m = ModelManifest.load(write_manifest(tmp_path, {"llama": {
            "url": srv.url + "/llama.zip", "sha256": sha(data), "size": len(data),
            "licence": "MIT", "required_by": ["llm.servers.local30b.exe"],
            "dest": "vendor/llama.cpp", "unpack": "zip", "variant": "cuda-13.4",
        }}), root=tmp_path)
        m.pull(["llama"])
        assert (tmp_path / "vendor" / "llama.cpp" / "llama-server.exe").read_bytes() == b"MZ fake"
        n = len(srv.requests)
        m.pull(["llama"])
        assert len(srv.requests) == n
        assert m.check("llama").ok
    finally:
        srv.close()


def test_zip_with_unsafe_paths_is_refused(tmp_path: Path) -> None:
    data = _zip({"../evil.txt": b"x"})
    srv = FileServer({"/bad.zip": data})
    try:
        m = ModelManifest.load(write_manifest(tmp_path, {"bad": {
            "url": srv.url + "/bad.zip", "sha256": sha(data), "size": len(data),
            "licence": "MIT", "required_by": [], "dest": "vendor/x", "unpack": "zip",
        }}), root=tmp_path)
        with pytest.raises(ModelError, match="unsafe"):
            m.pull(["bad"])
        assert not (tmp_path / "vendor" / "evil.txt").exists()
    finally:
        srv.close()


@pytest.mark.parametrize(
    "bad",
    [{"sha256": "abc"}, {"size": -1}, {"dest": "../outside"}, {"unpack": "tar"}, {"url": ""}],
)
def test_manifest_validation(tmp_path: Path, bad: dict[str, object]) -> None:
    fields: dict[str, object] = {"url": "https://x/y", "sha256": "", "size": 0, "licence": "MIT",
                                 "required_by": [], "dest": "models/y", **bad}
    with pytest.raises(ModelError):
        ModelManifest.load(write_manifest(tmp_path, {"y": fields}), root=tmp_path)


def test_cli_verify_and_list(tmp_path: Path, server: FileServer,
                             capsys: pytest.CaptureFixture[str]) -> None:
    manifest(tmp_path, server.url + "/blob.bin")
    assert M.main(["verify", "blob", "--root", str(tmp_path), "--variant", "cuda-13.4"]) == 1
    assert "missing" in capsys.readouterr().out
    assert M.main(["pull", "blob", "--root", str(tmp_path), "--variant", "cuda-13.4"]) == 0
    assert M.main(["verify", "blob", "--root", str(tmp_path), "--variant", "cuda-13.4"]) == 0
    assert "✔ blob" in capsys.readouterr().out
    assert M.main(["list", "--root", str(tmp_path), "--variant", "cuda-13.4"]) == 0
