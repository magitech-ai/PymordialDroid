"""Unit tests for platform-safe binary resolution in ``config.py``.

The legacy bundled directory contains Windows executables. These tests make
sure a stray Windows artifact cannot shadow a native binary or the downloader
on another platform, without changing Windows behaviour.
"""

import sys
from unittest.mock import patch

import pytest

from pymordialdroid import binary_fetch, config
from pymordialdroid.binary_fetch import BinaryDownloadError, UnsupportedPlatformError
from pymordialdroid.config import _bundled_or_downloaded, _is_native_binary

ELF = b"\x7fELF" + b"\x00" * 12
PE = b"MZ" + b"\x00" * 14
MACHO = b"\xcf\xfa\xed\xfe" + b"\x00" * 12


def test_is_native_binary_by_platform(tmp_path, monkeypatch):
    elf = tmp_path / "adb"
    elf.write_bytes(ELF)
    pe = tmp_path / "adb.exe"
    pe.write_bytes(PE)
    macho = tmp_path / "scrcpy"
    macho.write_bytes(MACHO)

    monkeypatch.setattr(sys, "platform", "linux")
    assert _is_native_binary(elf)
    assert not _is_native_binary(pe)
    assert not _is_native_binary(macho)

    monkeypatch.setattr(sys, "platform", "win32")
    assert _is_native_binary(pe)
    assert not _is_native_binary(elf)

    monkeypatch.setattr(sys, "platform", "darwin")
    assert _is_native_binary(macho)
    assert not _is_native_binary(elf)


def test_is_native_binary_missing_or_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    script = tmp_path / "script"
    script.write_bytes(b"#!/bin/sh\nexit 0\n")
    assert not _is_native_binary(tmp_path / "missing")
    assert not _is_native_binary(empty)
    assert _is_native_binary(script)


def test_is_native_binary_follows_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    target = tmp_path / "native-target"
    target.write_bytes(ELF)
    link = tmp_path / "native-link"
    link.symlink_to(target)
    assert _is_native_binary(link)


def test_windows_still_prefers_bundled_executable(tmp_path, monkeypatch):
    bundle = tmp_path / "bin"
    bundle.mkdir()
    exe = bundle / "adb.exe"
    exe.write_bytes(PE)
    monkeypatch.setattr(sys, "platform", "win32")
    with patch.object(
        binary_fetch, "ensure_binary", side_effect=AssertionError("must not download")
    ):
        assert _bundled_or_downloaded("adb.exe", "adb", bundle) == exe


def test_windows_artifact_does_not_shadow_downloader_on_linux(tmp_path, monkeypatch):
    bundle = tmp_path / "bin"
    bundle.mkdir()
    pe = bundle / "adb.exe"
    pe.write_bytes(PE)
    (bundle / "adb").symlink_to(pe)
    downloaded = tmp_path / "cache" / "adb"
    monkeypatch.setattr(sys, "platform", "linux")
    with (
        patch.object(binary_fetch, "BIN_DIR", tmp_path / "cache"),
        patch.object(binary_fetch, "is_binary_current", return_value=False),
        patch.object(binary_fetch, "ensure_binary", return_value=downloaded) as ensure,
    ):
        assert _bundled_or_downloaded("adb", "adb", bundle) == downloaded
        ensure.assert_called_once_with("adb")


def test_system_binary_fallback_on_unsupported_platform(tmp_path, monkeypatch):
    bundle = tmp_path / "empty"
    bundle.mkdir()
    system = tmp_path / "system" / "adb"
    monkeypatch.setattr(sys, "platform", "linux")
    with (
        patch.object(binary_fetch, "BIN_DIR", tmp_path / "cache"),
        patch.object(binary_fetch, "is_binary_current", return_value=False),
        patch.object(
            binary_fetch,
            "ensure_binary",
            side_effect=UnsupportedPlatformError("no pinned package"),
        ),
        patch.object(config.shutil, "which", return_value=str(system)),
    ):
        assert _bundled_or_downloaded("adb", "adb", bundle) == system


def test_system_binary_fallback_on_download_failure(tmp_path, monkeypatch):
    bundle = tmp_path / "empty"
    bundle.mkdir()
    system = tmp_path / "system" / "scrcpy"
    monkeypatch.setattr(sys, "platform", "linux")
    with (
        patch.object(binary_fetch, "BIN_DIR", tmp_path / "cache"),
        patch.object(binary_fetch, "is_binary_current", return_value=False),
        patch.object(
            binary_fetch,
            "ensure_binary",
            side_effect=BinaryDownloadError(
                "scrcpy", "url", bundle, OSError("offline")
            ),
        ),
        patch.object(config.shutil, "which", return_value=str(system)),
    ):
        assert _bundled_or_downloaded("scrcpy", "scrcpy", bundle) == system


def test_error_reraised_without_system_binary(tmp_path, monkeypatch):
    bundle = tmp_path / "empty"
    bundle.mkdir()
    monkeypatch.setattr(sys, "platform", "linux")
    with (
        patch.object(binary_fetch, "BIN_DIR", tmp_path / "cache"),
        patch.object(binary_fetch, "is_binary_current", return_value=False),
        patch.object(
            binary_fetch,
            "ensure_binary",
            side_effect=UnsupportedPlatformError("no pinned package"),
        ),
        patch.object(config.shutil, "which", return_value=None),
        pytest.raises(UnsupportedPlatformError),
    ):
        _bundled_or_downloaded("adb", "adb", bundle)
