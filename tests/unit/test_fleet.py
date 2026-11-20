"""Unit tests for FleetCommander in pymordialdroid.fleet."""

import asyncio
from pathlib import Path

import cv2
import numpy as np

from pymordialdroid.android_controller import AndroidController
from pymordialdroid.device import Phone
from pymordialdroid.fleet import FleetCommander
from pymordialdroid.models import DeviceRecord


def test_fleet_get_controller(tmp_path, mocker, sample_device_record):
    """Test retrieving controller by index or IP from FleetCommander."""
    fake_adb = tmp_path / "adb.exe"
    fake_adb.touch()
    fake_scrcpy = tmp_path / "scrcpy.exe"
    fake_scrcpy.touch()

    phone = Phone(
        record=sample_device_record,
        signer=mocker.MagicMock(),
        adb_path=fake_adb,
        scrcpy_path=fake_scrcpy,
    )

    fleet = FleetCommander()
    fleet.phones = [phone]

    controller = phone.controller
    assert isinstance(controller, AndroidController)

    assert fleet.get_controller(0) is controller
    assert fleet.get_controller("192.168.1.50") is controller
    assert fleet.get_controller("192.168.1.50:5555") is controller
    assert fleet.get_controller(99) is None
    assert fleet.get_controller("10.0.0.1") is None


def _make_mock_phone(
    tmp_path: Path,
    mocker,
    record: DeviceRecord,
    *,
    connected: bool = True,
    battery_level: int | None = 85,
    return_valid_png: bool = True,
    open_app_result: bool = True,
    close_app_result: bool = True,
    find_text_phrase_result: tuple[int, int] | None = None,
) -> Phone:
    """Build a Phone with mocked device/controller for fleet operation tests."""
    fake_adb = tmp_path / "adb.exe"
    fake_adb.touch()
    fake_scrcpy = tmp_path / "scrcpy.exe"
    fake_scrcpy.touch()

    phone = Phone(
        record=record,
        signer=mocker.MagicMock(),
        adb_path=fake_adb,
        scrcpy_path=fake_scrcpy,
    )
    phone._connected = connected
    phone.battery_level = battery_level
    phone.last_action = "test"

    # Generate a valid 1x1 red PNG for screenshot mocking
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:] = (0, 0, 255)  # red
    ok, buf = cv2.imencode(".png", img)
    valid_png = buf.tobytes() if ok else b""

    # Mock connect (async)
    phone.connect = mocker.AsyncMock(return_value=True)

    # Mock device.shell (async)
    phone.device = mocker.MagicMock()
    if return_valid_png:
        phone.device.shell = mocker.AsyncMock(return_value=valid_png)
    else:
        phone.device.shell = mocker.AsyncMock(return_value=None)

    # Mock controller methods (sync)
    phone.controller.open_app = mocker.Mock(return_value=open_app_result)
    phone.controller.close_app = mocker.Mock(return_value=close_app_result)
    phone.controller.ui.find_text_phrase = mocker.Mock(
        return_value=find_text_phrase_result
    )

    # Mock get_battery (async) - sets battery_level
    async def _fake_get_battery():
        phone.battery_level = battery_level

    mocker.patch.object(phone, "get_battery", side_effect=_fake_get_battery)

    return phone


def test_fleet_capture_screenshots(tmp_path, mocker, sample_device_record):
    """fleet_capture_screenshots captures PNG from each device."""
    phone = _make_mock_phone(tmp_path, mocker, sample_device_record)

    fleet = FleetCommander()
    fleet.phones = [phone]

    results = asyncio.run(fleet.fleet_capture_screenshots())

    assert sample_device_record.ip in results
    r = results[sample_device_record.ip]
    assert r.success is True
    assert r.detail == "captured"
    assert r.data is not None
    assert "png_bytes" in r.data
    assert "width" in r.data
    assert "height" in r.data


def test_fleet_capture_screenshots_failure(tmp_path, mocker, sample_device_record):
    """fleet_capture_screenshots returns failure when screencap fails."""
    phone = _make_mock_phone(
        tmp_path, mocker, sample_device_record, return_valid_png=False
    )

    fleet = FleetCommander()
    fleet.phones = [phone]

    results = asyncio.run(fleet.fleet_capture_screenshots())

    r = results[sample_device_record.ip]
    assert r.success is False


def test_fleet_find_text_phrase(tmp_path, mocker, sample_device_record):
    """fleet_find_text_phrase finds phrase coordinates on each device."""
    phone = _make_mock_phone(
        tmp_path, mocker, sample_device_record, find_text_phrase_result=(100, 200)
    )

    fleet = FleetCommander()
    fleet.phones = [phone]

    results = asyncio.run(fleet.fleet_find_text_phrase("Security and privacy"))

    r = results[sample_device_record.ip]
    assert r.success is True
    assert r.data == {"center_x": 100, "center_y": 200}


def test_fleet_find_text_phrase_not_found(tmp_path, mocker, sample_device_record):
    """fleet_find_text_phrase returns failure when phrase is absent."""
    phone = _make_mock_phone(
        tmp_path, mocker, sample_device_record, find_text_phrase_result=None
    )

    fleet = FleetCommander()
    fleet.phones = [phone]

    results = asyncio.run(fleet.fleet_find_text_phrase("NonExistent Phrase"))

    r = results[sample_device_record.ip]
    assert r.success is False


def test_fleet_open_app(tmp_path, mocker, sample_device_record):
    """fleet_open_app opens app on each device."""
    phone = _make_mock_phone(tmp_path, mocker, sample_device_record)

    fleet = FleetCommander()
    fleet.phones = [phone]

    results = asyncio.run(fleet.fleet_open_app("Settings", "com.android.settings"))

    r = results[sample_device_record.ip]
    assert r.success is True
    assert r.detail == "opened"
    phone.controller.open_app.assert_called_once_with(
        "Settings", package_name="com.android.settings"
    )


def test_fleet_close_app(tmp_path, mocker, sample_device_record):
    """fleet_close_app closes app on each device."""
    phone = _make_mock_phone(tmp_path, mocker, sample_device_record)

    fleet = FleetCommander()
    fleet.phones = [phone]

    results = asyncio.run(fleet.fleet_close_app("com.android.settings"))

    r = results[sample_device_record.ip]
    assert r.success is True
    assert r.detail == "closed"
    phone.controller.close_app.assert_called_once_with(
        package_name="com.android.settings"
    )


def test_fleet_device_health(tmp_path, mocker, sample_device_record):
    """fleet_device_health reports battery and connection per device."""
    phone = _make_mock_phone(tmp_path, mocker, sample_device_record)

    fleet = FleetCommander()
    fleet.phones = [phone]

    results = asyncio.run(fleet.fleet_device_health())

    health = results[sample_device_record.ip]
    assert health.ip == sample_device_record.ip
    assert health.name == sample_device_record.name
    assert health.connected is True
    assert health.battery_level == 85


def test_fleet_run_command(tmp_path, mocker, sample_device_record):
    """fleet_run_command executes shell command on each device."""
    phone = _make_mock_phone(tmp_path, mocker, sample_device_record)
    phone.device.shell = mocker.AsyncMock(return_value="root:/ #")

    fleet = FleetCommander()
    fleet.phones = [phone]

    results = asyncio.run(fleet.fleet_run_command("getprop ro.build.version.sdk"))

    r = results[sample_device_record.ip]
    assert r.success is True
    assert r.data == {"output": "root:/ #"}


def test_fleet_check_text_found(tmp_path, mocker, sample_device_record):
    """fleet_check_text returns success when phrase is visible."""
    phone = _make_mock_phone(
        tmp_path, mocker, sample_device_record, find_text_phrase_result=(100, 200)
    )

    fleet = FleetCommander()
    fleet.phones = [phone]

    results = asyncio.run(fleet.fleet_check_text("Connected devices"))

    r = results[sample_device_record.ip]
    assert r.success is True


def test_fleet_check_text_not_found(tmp_path, mocker, sample_device_record):
    """fleet_check_text returns failure when phrase is absent."""
    phone = _make_mock_phone(
        tmp_path, mocker, sample_device_record, find_text_phrase_result=None
    )

    fleet = FleetCommander()
    fleet.phones = [phone]

    results = asyncio.run(fleet.fleet_check_text("NonExistent Phrase"))

    r = results[sample_device_record.ip]
    assert r.success is False


def test_fleet_device_health_disconnected(tmp_path, mocker, sample_device_record):
    """fleet_device_health reports disconnected device as not connected."""
    phone = _make_mock_phone(tmp_path, mocker, sample_device_record, connected=False)

    fleet = FleetCommander()
    fleet.phones = [phone]

    results = asyncio.run(fleet.fleet_device_health())

    health = results[sample_device_record.ip]
    assert health.connected is False
