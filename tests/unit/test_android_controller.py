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


def test_controller_wait_for_element(mocker):
    """wait_for_element polls and returns True when element found."""
    controller = AndroidController()

    mock_screenshot = b"fake_png_bytes"
    mock_coords = (200, 400)
    controller.adb.capture_screenshot = mocker.Mock(return_value=mock_screenshot)

    # find_element returns None first time, then coords second time
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

    result = controller.wait_for_element(mocker.Mock(), timeout=5.0, interval=0.1)

    assert result is True
    assert calls[0] == 2
    time.sleep = real_sleep


def test_controller_wait_for_element_timeout(mocker):
    """wait_for_element returns False when element never appears."""
    controller = AndroidController()
    controller.adb.capture_screenshot = mocker.Mock(return_value=b"fake")
    controller.find_element = mocker.Mock(return_value=None)

    import time

    real_sleep = time.sleep
    time.sleep = mocker.Mock()  # don't actually sleep

    result = controller.wait_for_element(mocker.Mock(), timeout=1.0, interval=0.1)

    assert result is False
    time.sleep = real_sleep


def test_controller_swipe_to_find(mocker):
    """swipe_to_find swipes until element is found."""
    controller = AndroidController()

    mock_screenshot = b"fake_png_bytes"
    mock_coords = (200, 400)
    controller.adb.capture_screenshot = mocker.Mock(return_value=mock_screenshot)
    controller.adb.swipe = mocker.Mock(return_value=True)

    # find_element returns None first two times, then coords third time
    calls = [0]

    def fake_find(element, pymordial_screenshot=None, max_tries=1):
        calls[0] += 1
        if calls[0] >= 3:
            return mock_coords
        return None

    controller.find_element = mocker.Mock(side_effect=fake_find)

    import time

    real_sleep = time.sleep
    time.sleep = mocker.Mock()  # don't actually sleep

    result = controller.swipe_to_find(mocker.Mock(), max_swipes=5, interval=0.1)

    assert result == mock_coords
    assert calls[0] == 3
    # Swiped twice before finding on third attempt
    assert controller.adb.swipe.call_count == 2
    time.sleep = real_sleep


def test_controller_swipe_to_find_not_found(mocker):
    """swipe_to_find returns None when element never found."""
    controller = AndroidController()
    controller.adb.capture_screenshot = mocker.Mock(return_value=b"fake_png_bytes")
    controller.adb.swipe = mocker.Mock(return_value=True)
    controller.find_element = mocker.Mock(return_value=None)

    import time

    real_sleep = time.sleep
    time.sleep = mocker.Mock()  # don't actually sleep

    result = controller.swipe_to_find(mocker.Mock(), max_swipes=3, interval=0.1)

    assert result is None
    assert controller.adb.swipe.call_count == 3
    time.sleep = real_sleep


def test_controller_find_and_tap_with_retry(mocker):
    """find_and_tap_with_retry swipes then taps found element."""
    controller = AndroidController()

    mock_screenshot = b"fake_png_bytes"
    mock_coords = (200, 400)
    controller.adb.capture_screenshot = mocker.Mock(return_value=mock_screenshot)
    controller.adb.swipe = mocker.Mock(return_value=True)
    controller.adb.tap = mocker.Mock(return_value=True)

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

    result = controller.find_and_tap_with_retry(
        mocker.Mock(), times=2, max_swipes=5, interval=0.1
    )

    assert result is True
    assert calls[0] == 2
    controller.adb.swipe.assert_called_once()  # one swipe before finding
    controller.adb.tap.assert_called_once_with(mock_coords, times=2)
    time.sleep = real_sleep


def test_controller_find_and_tap_with_retry_not_found(mocker):
    """find_and_tap_with_retry returns False when element never found."""
    controller = AndroidController()
    controller.adb.capture_screenshot = mocker.Mock(return_value=b"fake_png_bytes")
    controller.adb.swipe = mocker.Mock(return_value=True)
    controller.adb.tap = mocker.Mock(return_value=True)
    controller.find_element = mocker.Mock(return_value=None)

    import time

    real_sleep = time.sleep
    time.sleep = mocker.Mock()  # don't actually sleep

    result = controller.find_and_tap_with_retry(
        mocker.Mock(), max_swipes=3, interval=0.1
    )

    assert result is False
    assert controller.adb.swipe.call_count == 3
    controller.adb.tap.assert_not_called()
    time.sleep = real_sleep


def test_controller_swipe_to_find_and_tap(mocker):
    """swipe_to_find_and_tap swipes then taps found element."""
    controller = AndroidController()

    mock_screenshot = b"fake_png_bytes"
    mock_coords = (200, 400)
    controller.adb.capture_screenshot = mocker.Mock(return_value=mock_screenshot)
    controller.adb.swipe = mocker.Mock(return_value=True)
    controller.adb.tap = mocker.Mock(return_value=True)

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

    result = controller.swipe_to_find_and_tap(
        mocker.Mock(), times=2, max_swipes=5, interval=0.1
    )

    assert result is True
    assert calls[0] == 2
    controller.adb.swipe.assert_called_once()  # one swipe before finding
    controller.adb.tap.assert_called_once_with(mock_coords, times=2)
    time.sleep = real_sleep


def test_controller_swipe_to_find_and_tap_not_found(mocker):
    """swipe_to_find_and_tap returns False when element never found."""
    controller = AndroidController()
    controller.adb.capture_screenshot = mocker.Mock(return_value=b"fake_png_bytes")
    controller.adb.swipe = mocker.Mock(return_value=True)
    controller.adb.tap = mocker.Mock(return_value=True)
    controller.find_element = mocker.Mock(return_value=None)

    import time

    real_sleep = time.sleep
    time.sleep = mocker.Mock()  # don't actually sleep

    result = controller.swipe_to_find_and_tap(mocker.Mock(), max_swipes=3, interval=0.1)

    assert result is False
    assert controller.adb.swipe.call_count == 3
    controller.adb.tap.assert_not_called()
    time.sleep = real_sleep


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
    """find_and_swipe finds element then swipes from its position."""
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
    # Swiped once to find element, once from found position
    assert controller.adb.swipe.call_count == 2
    controller.adb.swipe.assert_called_with(500, 500, 500, 500, 500)
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
