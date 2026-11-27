"""Install the optional NeuroHarmonics k-Wave Metal executable on Apple silicon."""

import argparse
import hashlib
import os
import platform
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

METAL_BINARY_NAME = "kspaceFirstOrder-Metal"
RELEASE = "v1.3.0-macos"
ARCHIVE_ROOT = f"{METAL_BINARY_NAME}-v1.3-macos-arm64"
URL = (
    "https://github.com/NeuroHarmonics/k-wave-cpp-mac/releases/download/"
    f"{RELEASE}/{ARCHIVE_ROOT}.tar.gz"
)
SHA256 = "c22b4c411dcb465fde36cf48d38df8411a8b83ee96610f254bf7770e879b0416"
_NOTICES = (
    "LICENSE",
    "License.md",
    "THIRD_PARTY_NOTICES.md",
    "licenses/FFTW.txt",
    "licenses/GPL-3.0.txt",
    "licenses/HDF5.txt",
    "licenses/LLVM-OpenMP.txt",
    "licenses/VkFFT.txt",
    "licenses/libaec.txt",
    "licenses/metal-cpp.txt",
)


def metal_supported() -> bool:
    """Return whether the current Python runs on native arm64 macOS 15 or newer."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    major = platform.mac_ver()[0].split(".")[0]
    return major.isdecimal() and int(major) >= 15


def metal_binary_path() -> Path:
    """Return the versioned path used by the installer and the k-Wave solver."""
    return (
        Path.home() / "Library/Caches/beamax/kwave-metal" / RELEASE / METAL_BINARY_NAME
    )


def _installed(binary: Path) -> bool:
    return (
        binary.is_file()
        and os.access(binary, os.X_OK)
        and all((binary.parent / name).is_file() for name in _NOTICES)
    )


def _download_release(destination: Path) -> None:
    with tempfile.TemporaryDirectory(dir=destination) as temporary:
        staging = Path(temporary)
        archive = staging / "release.tar.gz"
        try:
            with urlopen(URL, timeout=60) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Could not download k-Wave Metal from {URL}: {exc}. "
                "Retry with network access or pass binary_path to KWaveSolver "
                "to use an existing Metal executable."
            ) from exc
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != SHA256:
            raise RuntimeError(
                f"SHA-256 mismatch for {URL}: expected {SHA256}, got {digest}"
            )

        files = (*_NOTICES, METAL_BINARY_NAME)
        with tarfile.open(archive, "r:gz") as bundle:
            members = [bundle.getmember(f"{ARCHIVE_ROOT}/{name}") for name in files]
            if not all(member.isfile() for member in members):
                raise RuntimeError(
                    "Expected regular files in the k-Wave Metal release."
                )
            bundle.extractall(staging, members=members, filter="data")
        extracted = staging / ARCHIVE_ROOT
        (extracted / METAL_BINARY_NAME).chmod(0o755)
        (destination / "licenses").mkdir(exist_ok=True)
        # Publish each complete file atomically, with the executable installed last.
        for name in files:
            (extracted / name).replace(destination / name)


def install() -> Path:
    """Return the cached executable, downloading the verified release on first use."""
    if not metal_supported():
        raise RuntimeError(
            "k-Wave Metal requires native arm64 Python on macOS 15 or newer."
        )
    binary = metal_binary_path()
    if _installed(binary):
        return binary

    # fcntl is available on macOS. Other platforms can still import this module.
    import fcntl

    binary.parent.mkdir(parents=True, exist_ok=True)
    directory_fd = os.open(binary.parent, os.O_RDONLY)
    try:
        # Lock the release directory itself so interrupted installs leave no stale lock.
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        if not _installed(binary):
            _download_release(binary.parent)
    finally:
        os.close(directory_fd)
    return binary


def main() -> None:
    """Install the executable and print its absolute path."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        print(install())
    except (OSError, RuntimeError, tarfile.TarError) as exc:
        raise SystemExit(f"k-Wave Metal installation failed: {exc}") from exc


if __name__ == "__main__":
    main()
