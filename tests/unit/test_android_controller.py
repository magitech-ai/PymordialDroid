"""Unit tests for AndroidController in pymordialdroid."""

from pymordial.core.controller import PymordialController

from pymordialdroid.android_app import AndroidApp
from pymordialdroid.android_controller import AndroidController
from pymordialdroid.devices.adb_device import AdbDevice
from pymordialdroid.devices.scrcpy_device import ScrcpyDevice
from pymordialdroid.devices.tesseract_device import TesseractDevice
from pymordialdroid.devices.ui_device import AndroidUiDevice


def test_controller_satisfies_contract():
    """AndroidController must be a subclass of PymordialController."""
    assert issubclass(AndroidController, PymordialController)

    controller = AndroidController(
        ip="192.168.1.100",
        port=5555,
        device_name="TestDevice",
    )
    assert controller.ip == "192.168.1.100"
    assert controller.port == 5555
    assert controller.device_name == "TestDevice"
    assert isinstance(controller.bridge, AdbDevice)
    assert isinstance(controller.adb, AdbDevice)
    assert isinstance(controller.ui, AndroidUiDevice)
    assert isinstance(controller.scrcpy, ScrcpyDevice)
    assert isinstance(controller.ocr, TesseractDevice)


def test_controller_app_registration():
    """Registering an app with controller sets app.pymordial_controller."""
    app = AndroidApp(app_name="DemoApp", package_name="com.demo")
    controller = AndroidController()
    controller.add_app(app)

    assert "DemoApp" in controller.list_apps()
    assert app.pymordial_controller is controller


def test_controller_shutdown_stops_devices(mocker):
    """shutdown() stops scrcpy + adb and clears the frame provider."""
    controller = AndroidController()
    scrcpy = mocker.MagicMock()
    adb = mocker.MagicMock()
    controller.scrcpy = scrcpy
    controller.adb = adb

    controller.shutdown()

    scrcpy.shutdown.assert_called_once()
    adb.clear_frame_provider.assert_called_once()
    adb.shutdown.assert_called_once()


def test_controller_shutdown_tolerates_device_errors(mocker):
    """One device failing to shut down must not block the other."""
    controller = AndroidController()
    scrcpy = mocker.MagicMock()
    scrcpy.shutdown.side_effect = RuntimeError("scrcpy already gone")
    adb = mocker.MagicMock()
    controller.scrcpy = scrcpy
    controller.adb = adb

    controller.shutdown()  # must not raise

    adb.shutdown.assert_called_once()


def test_controller_ocr_and_extended_delegations(mocker):
    """Test controller delegates gestures, commands, and exposes OCR."""
    controller = AndroidController()
    assert controller.ocr is not None
    assert controller.ocr.name == "ocr"

    mocker.patch.object(controller.bridge, "swipe", return_value=True)
    assert controller.swipe(10, 20, 30, 40) is True
    controller.bridge.swipe.assert_called_with(
        start_x=10, start_y=20, end_x=30, end_y=40, duration=300
    )

    mocker.patch.object(controller.bridge, "go_back")
    controller.go_back()
    controller.bridge.go_back.assert_called_once()


def test_controller_find_and_tap_phrase(mocker):
    """find_and_tap_phrase captures screenshot, finds phrase, taps it."""
    controller = AndroidController()

    mock_screenshot = b"fake_png_bytes"
    mock_coords = (150, 300)
    controller.adb.capture_screenshot = mocker.Mock(return_value=mock_screenshot)
    controller.ocr.find_text_phrase = mocker.Mock(return_value=mock_coords)
    controller.adb.tap = mocker.Mock(return_value=True)

    result = controller.find_and_tap_phrase("Security and privacy", times=2)

    assert result is True
    controller.adb.capture_screenshot.assert_called_once()
    controller.ocr.find_text_phrase.assert_called_once_with(
        "Security and privacy", mock_screenshot, strategy=None, max_gap=1
    )
    controller.adb.tap.assert_called_once_with(mock_coords, times=2)


def test_controller_find_and_tap_phrase_not_found(mocker):
    """find_and_tap_phrase returns False when phrase is not found."""
    controller = AndroidController()
    controller.adb.capture_screenshot = mocker.Mock(return_value=b"fake")
    controller.ocr.find_text_phrase = mocker.Mock(return_value=None)

    result = controller.find_and_tap_phrase("NonExistent Phrase")

    assert result is False


def test_controller_wait_for_text(mocker):
    """wait_for_text polls screen until text appears."""
    controller = AndroidController()

    mock_screenshot = b"fake_png_bytes"
    controller.adb.capture_screenshot = mocker.Mock(return_value=mock_screenshot)

    # find_text returns None first time, then coords second time
    calls = [0]

    def fake_find_text(text_to_find, pymordial_screenshot=None, strategy=None):
        calls[0] += 1
        if calls[0] >= 2:
            return (200, 400)
        return None

    controller.ui.find_text = mocker.Mock(side_effect=fake_find_text)

    import time

    real_sleep = time.sleep
    time.sleep = mocker.Mock()  # don't actually sleep

    result = controller.wait_for_text("Connected devices", timeout=5.0, interval=0.1)

    assert result is True
    assert calls[0] == 2
    time.sleep = real_sleep


def test_controller_wait_for_text_timeout(mocker):
    """wait_for_text returns False when text never appears."""
    controller = AndroidController()
    controller.adb.capture_screenshot = mocker.Mock(return_value=b"fake")
    controller.ui.find_text = mocker.Mock(return_value=None)
    controller.ui.find_text_phrase = mocker.Mock(return_value=None)

    import time

    real_sleep = time.sleep
    time.sleep = mocker.Mock()  # don't actually sleep

    result = controller.wait_for_text("NonExistent", timeout=1.0, interval=0.1)

    assert result is False
    time.sleep = real_sleep


def test_controller_tap_text(mocker):
    """tap_text finds text and taps it."""
    controller = AndroidController()

    mock_screenshot = b"fake_png_bytes"
    mock_coords = (150, 300)
    controller.adb.capture_screenshot = mocker.Mock(return_value=mock_screenshot)
    controller.ocr.find_text_phrase = mocker.Mock(return_value=mock_coords)
    controller.adb.tap = mocker.Mock(return_value=True)

    result = controller.tap_text("Connected devices", times=2)

    assert result is True
    controller.adb.capture_screenshot.assert_called_once()
    controller.ocr.find_text_phrase.assert_called_once_with(
        "Connected devices", mock_screenshot, strategy=None, max_gap=1
    )
    controller.adb.tap.assert_called_once_with(mock_coords, times=2)


def test_controller_find_and_drag(mocker):
    """find_and_drag finds element then drags to target coordinates."""
    controller = AndroidController()

    mock_screenshot = b"fake_png_bytes"
    mock_coords = (100, 200)
    controller.adb.capture_screenshot = mocker.Mock(return_value=mock_screenshot)
    controller.adb.swipe = mocker.Mock(return_value=True)
    controller.adb.precise_drag = mocker.Mock(return_value=True)

    calls = [0]

    def fake_find(element, pymordial_screenshot=None, max_tries=1):
        calls[0] += 1
        if calls[0] >= 2:
            return mock_coords
        return None

    controller.find_element = mocker.Mock(side_effect=fake_find)

    import time

    real_sleep = time.sleep
    time.sleep = mocker.Mock()  # don't actually sleep

    result = controller.find_and_drag(
        mocker.Mock(), 400, 600, steps=5, step_delay_ms=10
    )

    assert result is True
    assert calls[0] == 2
    controller.adb.swipe.assert_called_once()  # one swipe before finding
    controller.adb.precise_drag.assert_called_once_with(
        100, 200, 400, 600, steps=5, step_delay_ms=10
    )
    time.sleep = real_sleep


def test_controller_find_and_swipe(mocker):
    """find_and_swipe finds element then swipes from its position (second call)."""
    controller = AndroidController()

    mock_screenshot = b"fake_png_bytes"
    mock_coords = (500, 1000)
    controller.adb.capture_screenshot = mocker.Mock(return_value=mock_screenshot)
    controller.adb.swipe = mocker.Mock(return_value=True)

    calls = [0]

    def fake_find(element, pymordial_screenshot=None, max_tries=1):
        calls[0] += 1
        if calls[0] >= 2:
            return mock_coords
        return None

    controller.find_element = mocker.Mock(side_effect=fake_find)

    import time

    real_sleep = time.sleep
    time.sleep = mocker.Mock()  # don't actually sleep

    result = controller.find_and_swipe(
        mocker.Mock(), direction="up", distance=500, max_swipes=5
    )

    assert result is True
    assert calls[0] == 2
    assert controller.adb.swipe.call_count == 2
    # Last call is the swipe from found position (500, 1000) up by 500 -> (500, 500)
    controller.adb.swipe.assert_called_with(500, 1000, 500, 500, 500)
    time.sleep = real_sleep


def test_controller_find_and_swipe_not_found(mocker):
    """find_and_swipe returns False when element never found."""
    controller = AndroidController()
    controller.adb.capture_screenshot = mocker.Mock(return_value=b"fake_png_bytes")
    controller.adb.swipe = mocker.Mock(return_value=True)
    controller.find_element = mocker.Mock(return_value=None)

    import time

    real_sleep = time.sleep
    time.sleep = mocker.Mock()  # don't actually sleep

    result = controller.find_and_swipe(mocker.Mock(), max_swipes=3, interval=0.1)

    assert result is False
    assert controller.adb.swipe.call_count == 3
    time.sleep = real_sleep


def test_controller_dismiss_common_dialogs(mocker):
    """dismiss_common_dialogs finds and taps dialog buttons."""
    controller = AndroidController()

    mock_screenshot = b"fake_png_bytes"
    controller.adb.capture_screenshot = mocker.Mock(return_value=mock_screenshot)
    controller.adb.tap = mocker.Mock(return_value=True)

    # First scan finds "Allow", second scan finds nothing
    call_count = [0]

    def fake_find_text_phrase(phrase, screenshot, strategy=None, max_gap=1):
        call_count[0] += 1
        if call_count[0] == 1:
            return (200, 400)  # Found "Allow"
        return None  # No more dialogs

    controller.ui.find_text_phrase = mocker.Mock(side_effect=fake_find_text_phrase)

    import time

    real_sleep = time.sleep
    time.sleep = mocker.Mock()  # don't actually sleep

    result = controller.dismiss_common_dialogs(max_attempts=3, interval=0.1)

    assert result == 1
    controller.adb.tap.assert_called_once_with((200, 400))
    time.sleep = real_sleep


def test_controller_dismiss_common_dialogs_no_dialogs(mocker):
    """dismiss_common_dialogs returns 0 when no dialogs found."""
    controller = AndroidController()

    controller.adb.capture_screenshot = mocker.Mock(return_value=b"fake_png_bytes")
    controller.adb.tap = mocker.Mock(return_value=True)
    controller.ui.find_text_phrase = mocker.Mock(return_value=None)

    import time

    real_sleep = time.sleep
    time.sleep = mocker.Mock()  # don't actually sleep

    result = controller.dismiss_common_dialogs(max_attempts=2, interval=0.1)

    assert result == 0
    controller.adb.tap.assert_not_called()
    time.sleep = real_sleep


def test_controller_record_and_replay_gesture(mocker):
    """record_gesture captures screenshots, replay_gesture taps them."""
    controller = AndroidController()

    # Create a fake screenshot (1x1 red pixel)
    import cv2
    import numpy as np

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:] = (255, 0, 0)  # Red
    _, buf = cv2.imencode(".png", img)
    fake_screenshot_bytes = buf.tobytes()

    controller.adb.capture_screenshot = mocker.Mock(return_value=fake_screenshot_bytes)
    controller.adb.tap = mocker.Mock(return_value=True)

    # Record
    events = controller.record_gesture(duration_seconds=0.1, sample_interval=0.05)
    assert len(events) > 0
    assert all("timestamp" in e for e in events)
    assert all("screenshot_b64" in e for e in events)
    assert all("change_score" in e for e in events)

    # Replay
    result = controller.replay_gesture(events, replay_speed=10.0)
    assert result is True
    controller.adb.tap.assert_called()


def test_controller_record_gesture_no_screenshot(mocker):
    """record_gesture handles missing screenshots gracefully."""
    controller = AndroidController()
    controller.adb.capture_screenshot = mocker.Mock(return_value=None)

    events = controller.record_gesture(duration_seconds=0.1, sample_interval=0.05)
    assert events == []


def test_controller_track_app_usage(mocker):
    """track_app_usage records app lifecycle events."""
    controller = AndroidController()

    # Mock get_current_app to return different apps over time
    apps = [
        "com.android.settings",
        "com.android.settings",
        "com.instagram.android",
        "com.instagram.android",
        None,
    ]
    calls = [0]

    def fake_get_current_app():
        calls[0] += 1
        idx = min(calls[0] - 1, len(apps) - 1)
        return apps[idx]

    controller.adb.get_current_app = mocker.Mock(side_effect=fake_get_current_app)

    import time

    real_sleep = time.sleep
    time.sleep = mocker.Mock()  # don't actually sleep

    results = controller.track_app_usage(duration_seconds=1.0, sample_interval=0.1)

    assert len(results) > 0
    # First record should be the first app
    assert results[0]["app"] == "com.android.settings"
    assert results[0]["state"] == "foreground"
    time.sleep = real_sleep


def test_controller_capture_on_change(mocker):
    """capture_on_change captures screenshots when screen changes."""
    controller = AndroidController()

    # Create two different screenshots
    import cv2
    import numpy as np

    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img1[:] = (255, 0, 0)  # Red
    _, buf1 = cv2.imencode(".png", img1)
    screenshot1 = buf1.tobytes()

    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2[:] = (0, 255, 0)  # Green
    _, buf2 = cv2.imencode(".png", img2)
    screenshot2 = buf2.tobytes()

    # Return screenshots in sequence, then repeat last one (no change triggers)
    screenshot_iter = iter([screenshot1, screenshot1, screenshot2, screenshot2])

    def mock_capture(*args, **kwargs):
        try:
            return next(screenshot_iter)
        except StopIteration:
            return screenshot2

    controller.adb.capture_screenshot = mocker.Mock(side_effect=mock_capture)

    import time

    real_sleep = time.sleep
    time.sleep = mocker.Mock()  # don't actually sleep

    results = controller.capture_on_change(timeout=5.0, interval=0.1, max_screenshots=5)

    assert len(results) >= 1
    # First capture always happens
    assert results[0]["change_score"] > 0
    # Second capture should detect change
    assert any(r["change_score"] > 0.1 for r in results)
    time.sleep = real_sleep


def test_controller_capture_on_change_no_change(mocker):
    """capture_on_change returns empty when screen doesn't change."""
    controller = AndroidController()

    # Same screenshot every time
    import cv2
    import numpy as np

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (255, 0, 0)
    _, buf = cv2.imencode(".png", img)
    screenshot_bytes = buf.tobytes()

    controller.adb.capture_screenshot = mocker.Mock(return_value=screenshot_bytes)

    import time

    real_sleep = time.sleep
    time.sleep = mocker.Mock()  # don't actually sleep

    results = controller.capture_on_change(timeout=1.0, interval=0.1, max_screenshots=5)

    # First capture always happens, subsequent ones should not trigger change
    assert len(results) == 1
    time.sleep = real_sleep


def test_controller_auto_heal_on_port_change_already_correct(mocker):
    """auto_heal_on_port_change returns True when port already matches and connected."""
    controller = AndroidController()
    controller.port = 5555
    controller.adb.is_connected = mocker.Mock(return_value=True)
    controller.adb._try_auto_heal_tcpip = mocker.Mock(return_value=False)

    result = controller.auto_heal_on_port_change(expected_port=5555)

    assert result is True
    controller.adb._try_auto_heal_tcpip.assert_not_called()


def test_controller_auto_heal_on_port_change_heals(mocker):
    """auto_heal_on_port_change heals when port matches but connection is dead."""
    controller = AndroidController()
    controller.port = 5555
    controller.adb.is_connected = mocker.Mock(return_value=False)
    controller.adb._try_auto_heal_tcpip = mocker.Mock(return_value=True)
    controller.adb.is_connected = mocker.Mock(return_value=True)

    import time

    real_sleep = time.sleep
    time.sleep = mocker.Mock()

    result = controller.auto_heal_on_port_change(expected_port=5555)

    assert result is True
    time.sleep = real_sleep


def test_controller_wait_for_screen_transition(mocker):
    """wait_for_screen_transition detects when from_text disappears and to_text appears."""
    controller = AndroidController()

    mock_screenshot = b"fake_png_bytes"
    controller.adb.capture_screenshot = mocker.Mock(return_value=mock_screenshot)

    # from_text visible first, then gone; to_text appears after
    call_count = [0]

    def fake_find_text(text_to_find, pymordial_screenshot=None, strategy=None):
        call_count[0] += 1
        if call_count[0] == 1:
            return (200, 400)  # from_text still visible
        if call_count[0] == 2:
            return None  # from_text gone
        if call_count[0] == 3:
            return (300, 500)  # to_text found
        return None

    controller.ui.find_text = mocker.Mock(side_effect=fake_find_text)

    import time

    real_sleep = time.sleep
    time.sleep = mocker.Mock()

    result = controller.wait_for_screen_transition(
        from_text="Old screen", to_text="New screen", timeout=10.0, interval=0.1
    )

    assert result is True
    time.sleep = real_sleep


def test_controller_wait_for_screen_transition_timeout(mocker):
    """wait_for_screen_transition returns False on timeout."""
    controller = AndroidController()
    controller.adb.capture_screenshot = mocker.Mock(return_value=b"fake_png_bytes")
    controller.ui.find_text = mocker.Mock(return_value=(100, 200))  # always found

    import time

    real_sleep = time.sleep
    time.sleep = mocker.Mock()

    result = controller.wait_for_screen_transition(
        from_text="Never leaves", to_text="Never arrives", timeout=1.0, interval=0.1
    )

    assert result is False
    time.sleep = real_sleep


def test_controller_read_text_caching(mocker):
    """read_text caches results by image content."""
    controller = AndroidController()

    screenshot_bytes = b"fake_png_bytes"
    ocr_result = ["Hello", "World"]

    controller.ui.read_text = mocker.Mock(return_value=ocr_result)

    # First call should invoke OCR
    result1 = controller.read_text(screenshot_bytes)
    assert result1 == ocr_result
    controller.ui.read_text.assert_called_once()

    # Second call with same bytes should return cached result
    result2 = controller.read_text(screenshot_bytes)
    assert result2 == ocr_result
    # read_text should still only be called once (cached)
    controller.ui.read_text.assert_called_once()


def test_controller_read_text_cache_different_bytes(mocker):
    """read_text calls OCR again for different image bytes."""
    controller = AndroidController()

    controller.ui.read_text = mocker.Mock(return_value=["Result"])

    controller.read_text(b"screenshot_a")
    controller.read_text(b"screenshot_b")

    assert controller.ui.read_text.call_count == 2


def test_controller_clear_ocr_cache(mocker):
    """clear_ocr_cache removes all cached entries."""
    controller = AndroidController()

    controller.ui.read_text = mocker.Mock(return_value=["Cached"])

    controller.read_text(b"screenshot_a")
    controller.read_text(b"screenshot_b")

    assert controller.ui.read_text.call_count == 2

    cleared = controller.clear_ocr_cache()
    assert cleared == 2

    # After clearing, OCR should be called again
    controller.read_text(b"screenshot_a")
    assert controller.ui.read_text.call_count == 3


def test_controller_recognize_swipe_pattern(mocker):
    """recognize_swipe_pattern detects swipe patterns from gesture events."""
    controller = AndroidController()

    import base64

    import cv2
    import numpy as np

    # Create two very different screenshots (high change score)
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img1[:] = (255, 0, 0)  # Red
    _, buf1 = cv2.imencode(".png", img1)
    screenshot1_b64 = base64.b64encode(buf1.tobytes()).decode("ascii")

    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2[:] = (0, 255, 0)  # Green
    _, buf2 = cv2.imencode(".png", img2)
    screenshot2_b64 = base64.b64encode(buf2.tobytes()).decode("ascii")

    # Events with high change scores (simulating a swipe)
    events = [
        {"timestamp": 0.0, "screenshot_b64": screenshot1_b64, "change_score": 0.0},
        {"timestamp": 0.1, "screenshot_b64": screenshot1_b64, "change_score": 0.0},
        {"timestamp": 0.2, "screenshot_b64": screenshot2_b64, "change_score": 0.9},
        {"timestamp": 0.3, "screenshot_b64": screenshot2_b64, "change_score": 0.8},
        {"timestamp": 0.4, "screenshot_b64": screenshot1_b64, "change_score": 0.0},
    ]

    patterns = controller.recognize_swipe_pattern(events)

    assert len(patterns) >= 1
    assert patterns[0]["direction"] in ("up", "down", "left", "right", "unknown")
    assert patterns[0]["duration_s"] > 0
    assert "max_change_score" in patterns[0]


def test_controller_recognize_swipe_pattern_empty(mocker):
    """recognize_swipe_pattern returns empty list for no events."""
    controller = AndroidController()
    patterns = controller.recognize_swipe_pattern([])
    assert patterns == []


def test_controller_app_usage_sessionize(mocker):
    """app_usage_sessionize groups app changes into sessions."""
    controller = AndroidController()

    records = [
        {"timestamp": 0.0, "app": "com.android.settings", "state": "foreground"},
        {"timestamp": 5.0, "app": "com.instagram.android", "state": "foreground"},
        {"timestamp": 10.0, "app": "com.android.settings", "state": "foreground"},
        # Gap > 30s starts new session
        {"timestamp": 45.0, "app": "com.whatsapp", "state": "foreground"},
        {"timestamp": 50.0, "app": "com.android.settings", "state": "foreground"},
    ]

    sessions = controller.app_usage_sessionize(records, session_gap_seconds=30.0)

    assert len(sessions) == 2
    # First session: 3 apps over 10 seconds
    assert sessions[0]["start_timestamp"] == 0.0
    assert sessions[0]["end_timestamp"] == 10.0
    assert "com.android.settings" in sessions[0]["apps"]
    assert "com.instagram.android" in sessions[0]["apps"]
    # Second session: 2 apps over 5 seconds
    assert sessions[1]["start_timestamp"] == 45.0
    assert sessions[1]["end_timestamp"] == 50.0
    assert "com.whatsapp" in sessions[1]["apps"]


def test_controller_app_usage_sessionize_empty(mocker):
    """app_usage_sessionize returns empty list for no records."""
    controller = AndroidController()
    sessions = controller.app_usage_sessionize([])
    assert sessions == []
