import hashlib
import io
import multiprocessing
import os
import tarfile
import time
from urllib.error import URLError

import pytest

from beamax import install_kwave_metal as installer


@pytest.fixture
def local_release(tmp_path, monkeypatch):
    pytest.importorskip("fcntl")
    binary = tmp_path / "cache" / installer.RELEASE / installer.METAL_BINARY_NAME
    archive = tmp_path / "release.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name in (*installer._NOTICES, installer.METAL_BINARY_NAME):
            contents = name.encode()
            entry = tarfile.TarInfo(f"{installer.ARCHIVE_ROOT}/{name}")
            entry.size = len(contents)
            bundle.addfile(entry, io.BytesIO(contents))

    monkeypatch.setattr(installer, "metal_supported", lambda: True)
    monkeypatch.setattr(installer, "metal_binary_path", lambda: binary)
    monkeypatch.setattr(installer, "URL", archive.as_uri())
    monkeypatch.setattr(
        installer, "SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    return archive, binary


def test_install_preserves_notices_and_reuses_cached_release(
    local_release, monkeypatch
):
    archive, binary = local_release
    assert installer.install() == binary
    assert binary.read_bytes() == installer.METAL_BINARY_NAME.encode()
    assert os.access(binary, os.X_OK)
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md", "licenses/metal-cpp.txt"):
        assert (binary.parent / name).read_bytes() == name.encode()

    archive.unlink()
    import fcntl

    monkeypatch.setattr(fcntl, "flock", lambda *args: pytest.fail("unexpected lock"))
    assert installer.install() == binary


def test_install_retries_after_checksum_failure(local_release, monkeypatch):
    _, binary = local_release
    with monkeypatch.context() as patch:
        patch.setattr(installer, "SHA256", "0" * 64)
        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            installer.install()
    assert not any(binary.parent.rglob("*"))
    assert installer.install() == binary


def test_offline_first_use_identifies_download_and_allows_retry(
    local_release, monkeypatch
):
    _, binary = local_release
    cause = URLError("offline")

    def unavailable(*args, **kwargs):
        raise cause

    with monkeypatch.context() as patch:
        patch.setattr(installer, "urlopen", unavailable)
        with pytest.raises(RuntimeError, match="binary_path") as error:
            installer.install()
    assert installer.URL in str(error.value)
    assert error.value.__cause__ is cause
    assert not any(binary.parent.rglob("*"))
    assert installer.install() == binary


def _install_in_process(archive, binary, digest, ready):
    # Spawned processes have independent copies of the installer module.
    def download(url, timeout):
        (archive.parent / f"download-{os.getpid()}").touch()
        time.sleep(0.15)
        return archive.open("rb")

    installer.metal_supported = lambda: True
    installer.metal_binary_path = lambda: binary
    installer.SHA256 = digest
    installer.urlopen = download
    ready.wait(timeout=30)
    assert installer.install() == binary


def test_concurrent_processes_download_once(local_release):
    archive, binary = local_release
    context = multiprocessing.get_context("spawn")
    ready = context.Barrier(2)
    workers = [
        context.Process(
            target=_install_in_process, args=(archive, binary, installer.SHA256, ready)
        )
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    try:
        for worker in workers:
            worker.join(timeout=45)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
    assert len(list(archive.parent.glob("download-*"))) == 1
    assert binary.read_bytes() == installer.METAL_BINARY_NAME.encode()


@pytest.mark.parametrize(
    ("system", "machine", "version", "supported"),
    [
        ("linux", "arm64", "", False),
        ("darwin", "x86_64", "15.0", False),
        ("darwin", "arm64", "14.7.1", False),
        ("darwin", "arm64", "15.0", True),
    ],
)
def test_host_support(system, machine, version, supported, monkeypatch):
    monkeypatch.setattr(installer.sys, "platform", system)
    monkeypatch.setattr(installer.platform, "machine", lambda: machine)
    monkeypatch.setattr(installer.platform, "mac_ver", lambda: (version, (), machine))
    assert installer.metal_supported() is supported


def test_unsupported_host_skips_download(monkeypatch):
    monkeypatch.setattr(installer, "metal_supported", lambda: False)
    monkeypatch.setattr(
        installer, "urlopen", lambda *a, **k: pytest.fail("unexpected download")
    )
    with pytest.raises(RuntimeError, match="macOS"):
        installer.install()
