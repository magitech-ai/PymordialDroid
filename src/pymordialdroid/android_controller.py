"""Primary Android device controller implementing Pymordial's PymordialController contract."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from pymordial.core.blueprints.extract_strategy import PymordialExtractStrategy
from pymordial.core.controller import PymordialController
from pymordial.core.plugin import PymordialPlugin
from pymordial.core.registry import PluginRegistry
from pymordial.ui.element import PymordialElement

from pymordialdroid.android_app import AndroidApp
from pymordialdroid.config import SystemConfig, resolve_system_config
from pymordialdroid.devices.adb_device import AdbDevice
from pymordialdroid.devices.scrcpy_device import ScrcpyDevice
from pymordialdroid.devices.tesseract_device import TesseractDevice
from pymordialdroid.devices.ui_device import AndroidUiDevice

log = logging.getLogger("pymordialdroid")


class AndroidController(PymordialController):
    """Main controller coordinating ADB bridge, Scrcpy viewer, UI vision, and OCR for an Android device."""

    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: int = 5555,
        device_name: str = "AndroidDevice",
        pin: str | None = None,
        apps: list[AndroidApp] | None = None,
        system_config: SystemConfig | None = None,
        touch_input_width: int | None = None,
        touch_input_height: int | None = None,
    ) -> None:
        super().__init__(apps=apps)  # type: ignore[arg-type]
        self.ip = ip
        self.port = port
        self.device_name = device_name
        self.pin = pin
        self.system_config = system_config or resolve_system_config()

        self.registry = PluginRegistry()

        # OCR result cache: key = (image_hash, case_sensitive, strategy_name)
        # value = list[str] of OCR text lines
        self._ocr_cache: dict[tuple[str, bool, str | None], list[str]] = {}

        # 1. Resolve ADB Device
        self.adb: AdbDevice = self._resolve_plugin(
            "adb",
            lambda: AdbDevice(
                host=self.ip,
                port=self.port,
                system_config=self.system_config,
                touch_input_width=touch_input_width,
                touch_input_height=touch_input_height,
            ),
        )
        self.bridge: AdbDevice = self.adb

        # 2. Resolve OCR Device
        self.ocr: TesseractDevice = self._resolve_plugin(
            "ocr",
            lambda: TesseractDevice(system_config=self.system_config),
        )

        # 3. Resolve UI Device
        def configure_ui(plugin: PymordialPlugin) -> None:
            if hasattr(plugin, "set_bridge_device"):
                plugin.set_bridge_device(self.adb)
            if hasattr(plugin, "set_ocr_device"):
                plugin.set_ocr_device(self.ocr)

        self.ui: AndroidUiDevice = self._resolve_plugin(
            "ui",
            lambda: AndroidUiDevice(bridge_device=self.adb, ocr_device=self.ocr),
            configure_found_plugin=configure_ui,
        )

        # 4. Resolve Scrcpy Device
        def configure_scrcpy(plugin: PymordialPlugin) -> None:
            if hasattr(plugin, "set_bridge_device"):
                plugin.set_bridge_device(self.adb)

        self.scrcpy: ScrcpyDevice = self._resolve_plugin(
            "scrcpy",
            lambda: ScrcpyDevice(
                bridge_adb=self.adb,
                ip=self.ip,
                port=self.port,
                device_name=self.device_name,
                pin=self.pin,
                system_config=self.system_config,
            ),
            configure_found_plugin=configure_scrcpy,
        )

        # Link controller to all registered apps
        if apps:
            for app in apps:
                app.pymordial_controller = self

    def _resolve_plugin(
        self,
        name: str,
        default_factory: Callable[[], PymordialPlugin],
        configure_found_plugin: Callable[[PymordialPlugin], None] | None = None,
    ) -> Any:
        """Resolves a plugin from the registry or creates a default."""
        try:
            plugin = self.registry.get(name)
            if configure_found_plugin:
                configure_found_plugin(plugin)
            return plugin
        except KeyError:
            default_plugin = default_factory()
            self.registry.register(default_plugin)
            return default_plugin

    def add_app(self, app: AndroidApp) -> None:
        """Registers an application with this controller."""
        super().add_app(app)
        app.pymordial_controller = self

    # --- PymordialController Abstract Implementations ---

    def capture_screen(self) -> bytes | None:
        """Captures the current screen, preferring live stream, then feed."""
        try:
            frame = self.scrcpy.get_latest_frame()
            if frame:
                return frame
        except Exception:
            pass
        return self.bridge.capture_screenshot()

    def click_coord(self, coords: tuple[int, int], times: int = 1) -> bool:
        """Clicks specific coordinates on the device screen."""
        return self.bridge.tap(coords, times=times)

    def tap(self, x: int, y: int, times: int = 1) -> bool:
        """Taps screen coordinates (x, y)."""
        return self.bridge.tap(coords=x, y=y, times=times)

    def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: int = 300,
    ) -> bool:
        """Performs a touch swipe gesture."""
        return self.bridge.swipe(
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
            duration=duration,
        )

    # --- Low-level touch injection (sendevent; thin delegation) ---

    @property
    def touch_available(self) -> bool:
        """True when low-level sendevent injection is usable on the device."""
        return self.bridge.touch_available

    def touch_down(self, x: float, y: float, slot: int = 0) -> bool:
        """Presses a contact down at (x, y) on the given multi-touch slot."""
        return self.bridge.touch_down(x, y, slot=slot)

    def touch_move(self, x: float, y: float, slot: int = 0) -> bool:
        """Moves an already-down contact to (x, y) on the given slot."""
        return self.bridge.touch_move(x, y, slot=slot)

    def touch_up(self, slot: int = 0) -> bool:
        """Lifts the contact on the given slot."""
        return self.bridge.touch_up(slot=slot)

    def touch_cancel(self, slot: int = 0) -> bool:
        """Cancels the active gesture on the given slot (recovery)."""
        return self.bridge.touch_cancel(slot=slot)

    def touch_hold(self, x: float, y: float, duration_ms: int, slot: int = 0) -> bool:
        """Holds a contact down at (x, y) for ``duration_ms``, then releases."""
        return self.bridge.touch_hold(x, y, duration_ms, slot=slot)

    def held_touch(self, x: float, y: float, slot: int = 0):  # type: ignore[no-untyped-def]
        """Context manager holding a contact down for the block's duration."""
        return self.bridge.held_touch(x, y, slot=slot)

    def precise_drag(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        steps: int = 24,
        step_delay_ms: int = 16,
        slot: int = 0,
    ) -> bool:
        """Drags (x1, y1) -> (x2, y2) emitting ``steps`` interpolated moves."""
        return self.bridge.precise_drag(
            x1, y1, x2, y2, steps=steps, step_delay_ms=step_delay_ms, slot=slot
        )

    def find_element(
        self,
        pymordial_element: PymordialElement,
        pymordial_screenshot: bytes | None = None,
        max_tries: int = 1,
    ) -> tuple[int, int] | None:
        """Finds the coordinates of a UI element on screen."""
        return self.ui.where_element(
            pymordial_element,
            screenshot=pymordial_screenshot,
            max_tries=max_tries,
        )

    def is_element_visible(
        self,
        pymordial_element: PymordialElement,
        pymordial_screenshot: bytes | None = None,
        max_tries: int | None = None,
    ) -> bool:
        """Checks if a UI element is visible on screen."""
        tries = max_tries if max_tries is not None else 1
        return (
            self.find_element(pymordial_element, pymordial_screenshot, max_tries=tries)
            is not None
        )

    def click_element(
        self,
        pymordial_element: PymordialElement,
        times: int = 1,
        screenshot_img_bytes: bytes | None = None,
        max_tries: int = 1,
    ) -> bool:
        """Finds and clicks a UI element on screen."""
        coords = self.find_element(
            pymordial_element,
            pymordial_screenshot=screenshot_img_bytes,
            max_tries=max_tries,
        )
        if coords is not None:
            return self.click_coord(coords, times=times)
        return False

    def click_elements(
        self,
        pymordial_elements: list[PymordialElement],
        screenshot_img_bytes: bytes | None = None,
        max_tries: int = 1,
    ) -> bool:
        """Clicks the first matching element from a list."""
        for element in pymordial_elements:
            if self.click_element(
                element,
                screenshot_img_bytes=screenshot_img_bytes,
                max_tries=max_tries,
            ):
                return True
        return False

    def find_and_tap(
        self,
        pymordial_element: PymordialElement,
        times: int = 1,
        max_tries: int = 1,
    ) -> bool:
        """Finds a UI element on screen and taps its center coordinates.

        Captures a fresh screenshot, locates the element using the UI device,
        and taps it through the ADB bridge. Combines vision + touch in one
        call — the core "see and act" primitive for device automation.

        Args:
            pymordial_element: The element to find
                (PymordialImage, PymordialText, or PymordialPixel).
            times: Number of taps to perform at the found coordinates.
            max_tries: Maximum detection attempts per screenshot.

        Returns:
            True if the element was found and tapped, False otherwise.

        Example:
            # Tap the "Connections" menu item in Settings
            from pymordial.ui.text import PymordialText
            element = PymordialText(
                label="Connections", element_text="Connections",
                og_resolution=(1080, 2340),
            )
            controller.find_and_tap(element)
        """
        screenshot = self.adb.capture_screenshot()
        if screenshot is None:
            return False
        coords = self.find_element(
            pymordial_element,
            pymordial_screenshot=screenshot,
            max_tries=max_tries,
        )
        if coords is not None:
            return self.click_coord(coords, times=times)
        return False

    def find_and_tap_phrase(
        self,
        phrase: str,
        times: int = 1,
        max_tries: int = 1,
    ) -> bool:
        """Finds a multi-word phrase via OCR and taps its center.

        Uses :meth:`TesseractDevice.find_text_phrase` to match phrases that
        Tesseract splits across multiple per-word boxes (e.g. "Security and
        privacy"). Captures a fresh screenshot, locates the phrase, and taps
        its combined bounding-box center.

        Args:
            phrase: Multi-word phrase to search for (case-insensitive).
            times: Number of taps to perform at the found coordinates.
            max_tries: Maximum detection attempts per screenshot.

        Returns:
            True if the phrase was found and tapped, False otherwise.

        Example:
            controller.find_and_tap_phrase("Security and privacy")
        """
        for _ in range(max_tries):
            screenshot = self.adb.capture_screenshot()
            if screenshot is None:
                return False
            coords = self.ocr.find_text_phrase(
                phrase, screenshot, strategy=None, max_gap=1
            )
            if coords is not None:
                return self.click_coord(coords, times=times)
        return False

    def wait_for_element(
        self,
        pymordial_element: PymordialElement,
        timeout: float = 10.0,
        interval: float = 0.5,
    ) -> bool:
        """Polls the screen until an element becomes visible.

        Captures a screenshot every ``interval`` seconds and runs element
        detection. Returns True as soon as the element is found, or False
        when the timeout expires. Useful for waiting for app launches,
        page navigations, or dialogs to appear.

        Args:
            pymordial_element: The element to wait for.
            timeout: Maximum seconds to wait.
            interval: Seconds between detection attempts.

        Returns:
            True when the element is found, False on timeout.
        """
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            screenshot = self.adb.capture_screenshot()
            if screenshot is None:
                time.sleep(interval)
                continue
            coords = self.find_element(
                pymordial_element,
                pymordial_screenshot=screenshot,
                max_tries=1,
            )
            if coords is not None:
                return True
            time.sleep(interval)
        return False

    def swipe_to_find(
        self,
        pymordial_element: PymordialElement,
        max_swipes: int = 5,
        swipe_distance: int = 1000,
        swipe_duration: int = 500,
        interval: float = 1.0,
    ) -> tuple[int, int] | None:
        """Scrolls the screen until an element is found.

        Captures a screenshot, searches for the element, and if not found
        swipes the screen and tries again. Useful for feeds, lists, and
        other scrollable content where the target element may not be
        immediately visible.

        Args:
            pymordial_element: The element to find.
            max_swipes: Maximum number of swipe attempts.
            swipe_distance: Distance to swipe in pixels (negative = up).
            swipe_duration: Duration of each swipe in milliseconds.
            interval: Seconds to wait between swipe attempts.

        Returns:
            (x, y) coordinates of the found element, or None if not found
            after max_swipes attempts.

        Example:
            # Scroll until "Like" button is visible, then tap it
            from pymordial.ui.text import PymordialText
            element = PymordialText(
                label="Like", element_text="Like", og_resolution=(1080, 2340),
            )
            coords = controller.swipe_to_find(element, max_swipes=10)
            if coords:
                controller.tap(coords)
        """
        import time

        for attempt in range(max_swipes):
            screenshot = self.adb.capture_screenshot()
            if screenshot is None:
                time.sleep(interval)
                continue
            coords = self.find_element(
                pymordial_element,
                pymordial_screenshot=screenshot,
                max_tries=1,
            )
            if coords is not None:
                return coords
            # Swipe up to reveal more content
            self.adb.swipe(500, 1600, 500, 600, swipe_duration)
            time.sleep(interval)
        return None

    def find_and_tap_with_retry(
        self,
        pymordial_element: PymordialElement,
        times: int = 1,
        max_swipes: int = 5,
        swipe_distance: int = 1000,
        interval: float = 1.0,
    ) -> bool:
        """Finds an element by scrolling if necessary, then taps it.

        Combines :meth:`swipe_to_find` with :meth:`click_coord`. Useful
        for elements that may not be immediately visible (e.g. posts in a
        feed, menu items below the fold).

        Args:
            pymordial_element: The element to find and tap.
            times: Number of taps to perform at the found coordinates.
            max_swipes: Maximum number of swipe attempts before giving up.
            swipe_distance: Distance to swipe in pixels.
            interval: Seconds to wait between swipe attempts.

        Returns:
            True if the element was found and tapped, False otherwise.

        Example:
            from pymordial.ui.text import PymordialText
            element = PymordialText(
                label="Like", element_text="Like", og_resolution=(1080, 2340),
            )
            controller.find_and_tap_with_retry(element, max_swipes=10)
        """
        coords = self.swipe_to_find(
            pymordial_element,
            max_swipes=max_swipes,
            swipe_distance=swipe_distance,
            interval=interval,
        )
        if coords is not None:
            return self.click_coord(coords, times=times)
        return False

    def swipe_to_find_and_tap(
        self,
        pymordial_element: PymordialElement,
        times: int = 1,
        max_swipes: int = 5,
        swipe_distance: int = 1000,
        swipe_duration: int = 500,
        interval: float = 1.0,
    ) -> bool:
        """Swipes until an element is found, then taps it.

        A convenience wrapper around :meth:`find_and_tap_with_retry`
        with a more descriptive name. Useful for scrollable content
        (feeds, lists, stories) where the target element may not be
        immediately visible.

        Args:
            pymordial_element: The element to find and tap.
            times: Number of taps to perform at the found coordinates.
            max_swipes: Maximum number of swipe attempts before giving up.
            swipe_distance: Distance to swipe in pixels.
            swipe_duration: Duration of each swipe in milliseconds.
            interval: Seconds to wait between swipe attempts.

        Returns:
            True if the element was found and tapped, False otherwise.

        Example:
            from pymordial.ui.text import PymordialText
            element = PymordialText(
                label="Like", element_text="Like",
                og_resolution=(1080, 2340),
            )
            controller.swipe_to_find_and_tap(element, max_swipes=10)
        """
        return self.find_and_tap_with_retry(
            pymordial_element,
            times=times,
            max_swipes=max_swipes,
            swipe_distance=swipe_distance,
            interval=interval,
        )

    def wait_for_text(
        self,
        text: str,
        timeout: float = 10.0,
        interval: float = 0.5,
        strategy: PymordialExtractStrategy | None = None,
    ) -> bool:
        """Waits for specific text to appear on screen.

        Captures a screenshot every ``interval`` seconds and runs OCR
        text detection. Returns True as soon as the text is found, or
        False when the timeout expires. Useful for waiting for app
        launches, page navigations, or modal dialogs to appear.

        Args:
            text: Text to wait for (case-insensitive substring match).
            timeout: Maximum seconds to wait.
            interval: Seconds between detection attempts.
            strategy: Optional OCR preprocessing strategy.

        Returns:
            True when the text is found, False on timeout.

        Example:
            controller.wait_for_text("Connected devices", timeout=10.0)
        """
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            screenshot = self.adb.capture_screenshot()
            if screenshot is None:
                time.sleep(interval)
                continue
            coords = self.ui.find_text(
                text, pymordial_screenshot=screenshot, strategy=strategy
            )
            if coords is not None:
                return True
            # Also try phrase matching for multi-word text
            if hasattr(self.ui, "find_text_phrase"):
                coords = self.ui.find_text_phrase(text, screenshot, strategy=strategy)
                if coords is not None:
                    return True
            time.sleep(interval)
        return False

    def wait_for_screen_transition(
        self,
        from_text: str | None = None,
        to_text: str | None = None,
        timeout: float = 30.0,
        interval: float = 0.5,
        strategy: PymordialExtractStrategy | None = None,
    ) -> bool:
        """Waits for the screen to transition from one state to another.

        Captures screenshots at regular intervals and runs OCR text
        detection. Returns True as soon as ``to_text`` is detected on
        screen. Optionally waits first for ``from_text`` to disappear
        (indicating the previous screen is gone) before looking for
        ``to_text``.

        This is useful for automation flows where you need to confirm
        that a navigation or state change has completed before proceeding
        (e.g., waiting for a settings page to appear after tapping a
        menu item).

        Args:
            from_text: Text that should disappear before the transition
                is considered complete. If None, only ``to_text`` is
                waited for.
            to_text: Text that should appear when the transition is
                complete. If None, only ``from_text`` disappearing is
                waited for.
            timeout: Maximum seconds to wait for the transition.
            interval: Seconds between detection attempts.
            strategy: Optional OCR preprocessing strategy.

        Returns:
            True when the transition is detected, False on timeout.

        Example:
            # Wait for Settings page to appear after tapping "Connections"
            controller.wait_for_screen_transition(
                from_text="Connected devices",
                to_text="Connections",
                timeout=10.0,
            )
        """
        import time

        deadline = time.monotonic() + timeout
        from_gone = from_text is None
        to_found = to_text is None

        while time.monotonic() < deadline:
            screenshot = self.adb.capture_screenshot()
            if screenshot is None:
                time.sleep(interval)
                continue

            if not from_gone:
                coords = self.ui.find_text(
                    from_text, pymordial_screenshot=screenshot, strategy=strategy
                )
                if coords is not None:
                    # from_text still visible — haven't transitioned yet
                    time.sleep(interval)
                    continue
                from_gone = True
                log.debug(f"from_text '{from_text}' disappeared")

            if not to_found:
                coords = self.ui.find_text(
                    to_text, pymordial_screenshot=screenshot, strategy=strategy
                )
                if coords is not None:
                    return True
                # Also try phrase matching for multi-word text
                if hasattr(self.ui, "find_text_phrase"):
                    coords = self.ui.find_text_phrase(
                        to_text, screenshot, strategy=strategy
                    )
                    if coords is not None:
                        return True

            time.sleep(interval)
        return False

    def tap_text(
        self,
        text: str,
        times: int = 1,
        max_tries: int = 1,
        strategy: PymordialExtractStrategy | None = None,
    ) -> bool:
        """Finds text on screen and taps its center.

        Captures a screenshot, runs OCR to find the text, and taps the
        found coordinates. A convenience wrapper around
        :meth:`find_and_tap_phrase` for single-word or multi-word text.

        Args:
            text: Text to find and tap (case-insensitive).
            times: Number of taps to perform at the found coordinates.
            max_tries: Maximum detection attempts per screenshot.
            strategy: Optional OCR preprocessing strategy.

        Returns:
            True if the text was found and tapped, False otherwise.

        Example:
            controller.tap_text("Connected devices")
        """
        return self.find_and_tap_phrase(text, times=times, max_tries=max_tries)

    def find_and_drag(
        self,
        element: PymordialElement,
        target_x: int,
        target_y: int,
        times: int = 1,
        max_swipes: int = 5,
        steps: int = 24,
        step_delay_ms: int = 16,
    ) -> bool:
        """Finds an element by scrolling if necessary, then drags to
        the specified coordinates.

        Combines :meth:`swipe_to_find` with a drag gesture. Useful for
        dragging UI elements (sliders, scroll handles, drag handles)
        to a target position.

        Args:
            element: The element to find and drag.
            target_x: Target X coordinate.
            target_y: Target Y coordinate.
            times: Number of drag operations to perform.
            max_swipes: Maximum number of swipe attempts to find element.
            steps: Number of interpolated move events per drag.
            step_delay_ms: Delay between move events in milliseconds.

        Returns:
            True if the element was found and dragged, False otherwise.

        Example:
            element = PymordialText(
                label="Slider handle", element_text="Slider",
                og_resolution=(1080, 2340),
            )
            controller.find_and_drag(element, 800, 1200)
        """
        coords = self.swipe_to_find(
            element,
            max_swipes=max_swipes,
            swipe_distance=1000,
            interval=1.0,
        )
        if coords is None:
            return False
        for _ in range(times):
            if not self.adb.precise_drag(
                coords[0],
                coords[1],
                target_x,
                target_y,
                steps=steps,
                step_delay_ms=step_delay_ms,
            ):
                return False
        return True

    def find_and_swipe(
        self,
        element: PymordialElement,
        direction: str = "up",
        distance: int = 1000,
        max_swipes: int = 5,
        swipe_duration: int = 500,
        interval: float = 1.0,
    ) -> bool:
        """Finds an element by scrolling if necessary, then swipes from
        its location in the given direction.

        Args:
            element: The element to find and swipe from.
            direction: Direction to swipe ("up", "down", "left", "right").
            distance: Distance to swipe in pixels.
            max_swipes: Maximum number of swipe attempts to find element.
            swipe_duration: Duration of each swipe in milliseconds.
            interval: Seconds to wait between swipe attempts.

        Returns:
            True if the element was found and swiped from, False otherwise.

        Example:
            element = PymordialText(
                label="Card", element_text="Card title",
                og_resolution=(1080, 2340),
            )
            controller.find_and_swipe(element, direction="up", distance=800)
        """
        coords = self.swipe_to_find(
            element,
            max_swipes=max_swipes,
            swipe_distance=1000,
            interval=interval,
        )
        if coords is None:
            return False

        x, y = coords
        if direction == "up":
            target_y = max(0, y - distance)
            target_x = x
        elif direction == "down":
            target_y = y + distance
            target_x = x
        elif direction == "left":
            target_x = max(0, x - distance)
            target_y = y
        elif direction == "right":
            target_x = x + distance
            target_y = y
        else:
            log.warning(f"Unknown direction '{direction}', defaulting to up")
            target_x = x
            target_y = max(0, y - distance)

        return self.adb.swipe(x, y, target_x, target_y, swipe_duration)

    # --- DIALOG DISMISSAL ---

    def dismiss_common_dialogs(
        self,
        max_attempts: int = 3,
        interval: float = 1.0,
    ) -> int:
        """Dismiss common dialogs (permissions, updates, terms, etc.).

        Scans the screen for common dialog buttons and taps them.
        Repeats until no dialog buttons are found or max_attempts is reached.

        Args:
            max_attempts: Maximum number of scan-and-dismiss attempts.
            interval: Seconds to wait between attempts.

        Returns:
            Number of dialogs dismissed.

        Example:
            controller.dismiss_common_dialogs(max_attempts=5)
        """
        import time

        dismissed = 0
        dismiss_phrases = [
            "allow",
            "don't allow",
            "dont allow",
            "not now",
            "skip",
            "ok",
            "okay",
            "cancel",
            "agree",
            "accept",
            "install",
            "update",
            "later",
            "remind me later",
            "maybe later",
            "no thanks",
            "no, thanks",
            "continue",
            "got it",
            "got it!",
            "understood",
            "i understand",
        ]

        for attempt in range(max_attempts):
            screenshot = self.adb.capture_screenshot()
            if screenshot is None:
                time.sleep(interval)
                continue

            found_any = False
            for phrase in dismiss_phrases:
                coords = self.ui.find_text_phrase(phrase, screenshot)
                if coords is not None:
                    self.adb.tap(coords)
                    dismissed += 1
                    found_any = True
                    time.sleep(1.0)  # Wait for dialog to dismiss
                    break  # Rescan after each dismissal

            if not found_any:
                break

            time.sleep(interval)

        return dismissed

    # --- GESTURE RECORD & REPLAY ---

    def record_gesture(
        self,
        duration_seconds: float = 10.0,
        sample_interval: float = 0.1,
    ) -> list[dict]:
        """Record touch gestures by capturing screen changes.

        Since we cannot directly intercept touch events on a non-rooted
        device, this method records screen frames at regular intervals
        and detects significant changes between frames. Each detected
        change is recorded as a potential touch interaction point.

        Args:
            duration_seconds: How long to record in seconds.
            sample_interval: Seconds between screen captures.

        Returns:
            List of gesture events, each with:
                - timestamp: Seconds since recording start
                - screenshot: Base64-encoded PNG of the frame
                - change_score: How much the screen changed (0.0 to 1.0)

        Example:
            gestures = controller.record_gesture(duration_seconds=5.0)
            controller.replay_gesture(gestures)
        """
        import base64
        import time

        events = []
        start_time = time.monotonic()
        previous_frame = None

        while time.monotonic() - start_time < duration_seconds:
            screenshot = self.adb.capture_screenshot()
            if screenshot is None:
                time.sleep(sample_interval)
                continue

            timestamp = time.monotonic() - start_time
            screenshot_b64 = base64.b64encode(screenshot).decode("ascii")

            change_score = 0.0
            if previous_frame is not None:
                change_score = self._compute_frame_difference(
                    previous_frame, screenshot
                )

            events.append(
                {
                    "timestamp": timestamp,
                    "screenshot_b64": screenshot_b64,
                    "change_score": change_score,
                }
            )

            previous_frame = screenshot
            time.sleep(sample_interval)

        return events

    def replay_gesture(
        self,
        events: list[dict],
        replay_speed: float = 1.0,
    ) -> bool:
        """Replay a recorded gesture sequence.

        Replays recorded gestures by waiting the same amount of time
        between events and tapping at the center of changed regions.

        Args:
            events: List of gesture events from record_gesture().
            replay_speed: Speed multiplier (1.0 = real time, 2.0 = 2x fast).

        Returns:
            True if all gestures were replayed successfully.

        Example:
            gestures = controller.record_gesture(duration_seconds=5.0)
            controller.replay_gesture(gestures, replay_speed=2.0)
        """
        import base64
        import time

        import cv2
        import numpy as np

        if not events:
            return False

        start_time = time.monotonic()

        for i, event in enumerate(events):
            timestamp = event["timestamp"]
            target_time = start_time + timestamp / replay_speed

            # Wait until it's time for this event
            while time.monotonic() < target_time:
                time.sleep(0.01)

            # Decode screenshot and find changed regions
            screenshot_bytes = base64.b64decode(event["screenshot_b64"])
            nparr = np.frombuffer(screenshot_bytes, np.uint8)
            screenshot = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if screenshot is None:
                continue

            change_score = event.get("change_score", 0.0)
            # Tap on the first event, or on subsequent events with
            # significant change
            if i == 0 or change_score > 0.1:
                h, w = screenshot.shape[:2]
                self.adb.tap(w // 2, h // 2)
                time.sleep(0.5)

        return True

    def _compute_frame_difference(
        self,
        frame1: bytes | np.ndarray,
        frame2: bytes | np.ndarray,
    ) -> float:
        """Compute difference score between two frames (0.0 to 1.0).

        Returns 0.0 if identical, 1.0 if completely different.
        """
        import cv2
        import numpy as np

        # Decode if bytes
        if isinstance(frame1, bytes):
            nparr = np.frombuffer(frame1, np.uint8)
            frame1 = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if isinstance(frame2, bytes):
            nparr = np.frombuffer(frame2, np.uint8)
            frame2 = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        # Resize to same size if needed
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

        # Compute absolute difference
        diff = cv2.absdiff(frame1, frame2)
        mean_diff = np.mean(diff)

        # Normalize to 0.0-1.0
        return min(1.0, mean_diff / 255.0)

    # --- APP USAGE TRACKING ---

    def track_app_usage(
        self,
        duration_seconds: float = 60.0,
        sample_interval: float = 5.0,
    ) -> list[dict]:
        """Track which apps are running over time.

        Periodically checks the foreground app and records usage
        statistics. Useful for understanding app usage patterns.

        Args:
            duration_seconds: How long to track in seconds.
            sample_interval: Seconds between checks.

        Returns:
            List of usage records, each with:
                - timestamp: Seconds since tracking start
                - app: Current foreground app package name
                - state: 'foreground' or 'background'

        Example:
            usage = controller.track_app_usage(duration_seconds=30.0,
                                              interval=2.0)
            for record in usage:
                print(f"{record['timestamp']:.1f}s: {record['app']}")
        """
        import time

        records = []
        start_time = time.monotonic()
        previous_app = None

        while time.monotonic() - start_time < duration_seconds:
            current_app = self.get_current_app()
            timestamp = time.monotonic() - start_time

            if current_app != previous_app:
                records.append(
                    {
                        "timestamp": timestamp,
                        "app": current_app,
                        "state": "foreground" if current_app else "background",
                    }
                )
                previous_app = current_app

            time.sleep(sample_interval)

        return records

    # --- SCREEN CHANGE CAPTURE ---

    def capture_on_change(
        self,
        timeout: float = 30.0,
        interval: float = 0.5,
        max_screenshots: int = 10,
        change_threshold: float = 0.1,
    ) -> list[dict]:
        """Capture screenshots when the screen changes significantly.

        Captures screenshots at regular intervals and saves them only
        when the screen content changes by more than the threshold.
        Useful for capturing state changes without filling storage
        with identical frames.

        Args:
            timeout: Maximum seconds to wait for changes.
            interval: Seconds between screenshot checks.
            max_screenshots: Maximum number of screenshots to capture.
            change_threshold: Minimum change score (0.0 to 1.0) to
                trigger a capture.

        Returns:
            List of captured screenshots, each with:
                - timestamp: Seconds since start
                - screenshot_b64: Base64-encoded PNG
                - change_score: How much changed (0.0 to 1.0)

        Example:
            captures = controller.capture_on_change(
                timeout=30.0,
                max_screenshots=5,
            )
            for cap in captures:
                print(f"Captured at {cap['timestamp']:.1f}s "
                      f"(change: {cap['change_score']:.2f})")
        """
        import base64
        import time

        captures = []
        start_time = time.monotonic()
        previous_frame = None

        while time.monotonic() - start_time < timeout:
            if len(captures) >= max_screenshots:
                break

            screenshot = self.adb.capture_screenshot()
            if screenshot is None:
                time.sleep(interval)
                continue

            timestamp = time.monotonic() - start_time
            change_score = 0.0

            if previous_frame is not None:
                change_score = self._compute_frame_difference(
                    previous_frame, screenshot
                )
            else:
                # First capture always counts
                change_score = 1.0

            if change_score >= change_threshold:
                screenshot_b64 = base64.b64encode(screenshot).decode("ascii")
                captures.append(
                    {
                        "timestamp": timestamp,
                        "screenshot_b64": screenshot_b64,
                        "change_score": change_score,
                    }
                )

            previous_frame = screenshot
            time.sleep(interval)

        return captures

    def shutdown(self) -> None:
        """Stops this controller's devices and releases their resources.

        Terminates the scrcpy viewer, live stream, and any headless
        recording, clears the ADB frame provider, and disconnects ADB.
        Best-effort per device: one device failing to shut down must not
        block the other.

        Called when this controller is retired — e.g. :class:`Phone`
        rebuilds its controller after an auto-heal port change. Without
        it, the retired controller's live stream and recording keep running
        with no handle left to stop them.
        """
        scrcpy = getattr(self, "scrcpy", None)
        if scrcpy is not None:
            try:
                scrcpy.shutdown()
            except Exception as e:
                log.debug(f"Error shutting down scrcpy device: {e}")
        adb = getattr(self, "adb", None)
        if adb is not None:
            try:
                if hasattr(adb, "clear_frame_provider"):
                    adb.clear_frame_provider()
            except Exception:
                pass
            try:
                adb.shutdown()
            except Exception as e:
                log.debug(f"Error shutting down adb device: {e}")

    def auto_heal_on_port_change(
        self,
        expected_port: int = 5555,
        timeout_s: float = 15.0,
    ) -> bool:
        """Attempt to auto-heal the ADB connection if the port has changed.

        Android devices sometimes switch their ADB port after a reboot,
        Wi-Fi reconnect, or when scrcpy releases the dynamic port. This
        method delegates to the underlying :class:`AdbDevice` auto-heal
        logic (scanning attached devices via `adb devices -l`, enabling
        tcpip on the matching serial, and reconnecting on port 5555).

        Args:
            expected_port: The port we expect the device to be on. If the
                current port already matches, returns True immediately.
            timeout_s: Maximum seconds to spend attempting auto-heal.

        Returns:
            True if the connection is healthy (either already correct or
            successfully healed), False otherwise.

        Example:
            if not controller.auto_heal_on_port_change():
                log.warning("Could not heal ADB connection")
        """
        import time

        if self.port == expected_port:
            # Quick health check before declaring success
            if self.adb.is_connected():
                return True
            # Port matches but connection is dead — try to heal anyway

        deadline = time.monotonic() + timeout_s

        # Try the built-in auto-heal (scans attached devices, enables tcpip)
        try:
            if self.adb._try_auto_heal_tcpip():
                # Re-verify the connection
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(min(remaining, 2.0))
                return self.adb.is_connected()
        except Exception as e:
            log.debug(f"AdbDevice auto-heal raised: {e}")

        # Fallback: try a direct reconnect on the expected port
        try:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                self.adb.disconnect()
                self.port = expected_port
                return self.adb.connect(auto_heal=False)
        except Exception as e:
            log.debug(f"Direct reconnect on port {expected_port} failed: {e}")

        return False

    def open_app(
        self,
        app_name: str | AndroidApp,
        package_name: str | None = None,
        timeout: int = 60,
        wait_time: int = 2,
    ) -> bool:
        """Opens an Android application on the device."""
        if isinstance(app_name, AndroidApp):
            pkg = package_name or getattr(app_name, "package_name", None)
            display_name = getattr(
                app_name, "name", getattr(app_name, "app_name", str(app_name))
            )
        else:
            pkg = package_name
            display_name = app_name

        return self.bridge.open_app(
            package_name=pkg or "",
            app_name=display_name,
            timeout=timeout,
            wait_time=wait_time,
        )

    def close_app(
        self,
        package_name: str | AndroidApp | None = None,
        app_name: str | None = None,
        timeout: int = 30,
        wait_time: int = 1,
    ) -> bool:
        """Closes an Android application on the device."""
        if isinstance(package_name, AndroidApp):
            pkg = getattr(package_name, "package_name", None)
            name = getattr(
                package_name,
                "name",
                getattr(package_name, "app_name", None),
            )
        else:
            pkg = package_name
            name = app_name

        return self.bridge.close_app(
            package_name=pkg,
            app_name=name,
            timeout=timeout,
            wait_time=wait_time,
        )

    def close_all_apps(self, exclude: list[str] | None = None) -> int:
        """Force stops all third-party apps."""
        return self.bridge.close_all_apps(exclude=exclude)

    def get_current_app(self) -> str | None:
        """Returns currently focused app package name."""
        return self.bridge.get_current_app()

    def read_text(
        self,
        image_path: Path | bytes | str,
        case_sensitive: bool = False,
        strategy: PymordialExtractStrategy | None = None,
    ) -> list[str]:
        """Reads text lines from image using OCR.

        Results are cached by image content hash so that repeated calls
        with the same screenshot (and same case_sensitive/strategy
        settings) return the cached result without re-running OCR.

        Args:
            image_path: Path to image, raw bytes, or string path.
            case_sensitive: Whether OCR should be case-sensitive.
            strategy: Optional OCR preprocessing strategy.

        Returns:
            List of text lines detected in the image.
        """
        import hashlib

        # Build a cache key from image content + options
        if isinstance(image_path, (bytes, bytearray)):
            image_hash = hashlib.sha256(bytes(image_path)).hexdigest()
        elif isinstance(image_path, str):
            try:
                from pathlib import Path as _Path

                p = _Path(image_path)
                if p.exists():
                    image_hash = hashlib.sha256(p.read_bytes()).hexdigest()
                else:
                    image_hash = hashlib.sha256(image_path.encode()).hexdigest()
            except Exception:
                image_hash = hashlib.sha256(image_path.encode()).hexdigest()
        else:
            # Path object
            try:
                image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
            except Exception:
                image_hash = str(hash(image_path))

        strategy_name = strategy.name if hasattr(strategy, "name") else str(strategy)
        cache_key = (image_hash, case_sensitive, strategy_name)

        if cache_key in self._ocr_cache:
            return self._ocr_cache[cache_key]

        result = self.ui.read_text(
            image_path, case_sensitive=case_sensitive, strategy=strategy
        )

        self._ocr_cache[cache_key] = result
        return result

    def check_text(
        self,
        text_to_find: str,
        image_path: Path | bytes | str,
        case_sensitive: bool = False,
        strategy: PymordialExtractStrategy | None = None,
    ) -> bool:
        """Checks if text exists in image using OCR.

        Results are cached by image content hash (same as
        :meth:`read_text`), so repeated checks on the same screenshot
        avoid re-running OCR.

        Args:
            text_to_find: Text to search for.
            image_path: Path to image, raw bytes, or string path.
            case_sensitive: Whether OCR should be case-sensitive.
            strategy: Optional OCR preprocessing strategy.

        Returns:
            True if the text is found in the image, False otherwise.
        """
        lines = self.read_text(
            image_path, case_sensitive=case_sensitive, strategy=strategy
        )
        if case_sensitive:
            return text_to_find in lines
        return any(text_to_find.lower() in line.lower() for line in lines)

    def clear_ocr_cache(self) -> int:
        """Clears the OCR result cache and returns the number of entries removed."""
        count = len(self._ocr_cache)
        self._ocr_cache.clear()
        return count

    # --- Convenience Device Operations ---

    def go_home(self) -> None:
        """Presses device home button."""
        self.bridge.go_home()

    def go_back(self) -> None:
        """Presses device back button."""
        self.bridge.go_back()

    def press_enter(self) -> None:
        """Presses Enter key."""
        self.bridge.press_enter()

    def press_esc(self) -> None:
        """Presses Escape/Back key."""
        self.bridge.press_esc()

    def run_command(self, command: str, decode: bool = True) -> str | bytes | None:
        """Executes raw shell command on device."""
        return self.bridge.run_command(command, decode=decode)

    def type_text(self, text: str, enter: bool = False) -> bool:
        """Types text on the device."""
        return self.bridge.type_text(text, enter=enter)

    def open_viewer(
        self,
        rank: int = 0,
        ghost: bool = False,
        new_display: str | None = None,
        start_app: str | None = None,
        show_touches: bool = False,
    ) -> bool:
        """Opens scrcpy viewer window (optionally on a virtual display)."""
        return self.scrcpy.open(
            rank=rank,
            ghost=ghost,
            new_display=new_display,
            start_app=start_app,
            show_touches=show_touches,
        )

    def open_virtual_display(
        self,
        rank: int = 0,
        display: str = "1920x1080",
        start_app: str | None = None,
    ) -> bool:
        """Opens a viewer on an isolated virtual display (Android 10+)."""
        return self.scrcpy.open_virtual_display(
            rank=rank, display=display, start_app=start_app
        )

    def close_viewer(self) -> bool:
        """Closes scrcpy viewer window."""
        return self.scrcpy.close()

    # --- Headless recording (Feature 2) ---

    def start_recording(
        self,
        output_path: str | Path | None = None,
        show_touches: bool = False,
        time_limit: int | None = None,
        record_orientation: int | None = None,
    ) -> Path | None:
        """Starts headless MP4 evidence recording (no window)."""
        return self.scrcpy.start_recording(
            output_path=output_path,
            show_touches=show_touches,
            time_limit=time_limit,
            record_orientation=record_orientation,
        )

    def stop_recording(self) -> Path | None:
        """Stops headless recording, returning the output path (if any)."""
        return self.scrcpy.stop_recording()

    def is_recording(self) -> bool:
        """Checks if headless recording is running."""
        return self.scrcpy.is_recording()

    # --- Headless live feed (Feature 1) ---

    def start_headless_feed(
        self,
        output_path: str | Path | None = None,
        max_size: int = 1024,
        max_fps: int = 30,
        bit_rate: str = "2M",
        new_display: str | None = None,
        start_app: str | None = None,
        timeout: float = 30.0,
    ) -> bool:
        """Starts a headless live stream and routes screenshots through it.

        Delegates to the live stream (direct scrcpy-server H.264 via PyAV).
        The legacy MKV file feed was removed because scrcpy cannot decode
        frames without a display surface when ``--no-window`` is used.
        """
        ok = self.scrcpy.start_headless_feed(
            output_path=output_path,
            max_size=max_size,
            max_fps=max_fps,
            bit_rate=bit_rate,
            new_display=new_display,
            start_app=start_app,
            timeout=timeout,
        )
        if ok and hasattr(self.bridge, "set_frame_provider"):
            try:
                self.bridge.set_frame_provider(self.scrcpy.get_latest_frame)
            except Exception:
                pass
        return ok

    def stop_headless_feed(self) -> bool:
        """Stops the headless feed and restores plain screencap captures."""
        if hasattr(self.bridge, "clear_frame_provider"):
            try:
                # Only clear if the live stream is not still providing frames.
                if not self.scrcpy.is_live_running():
                    self.bridge.clear_frame_provider()
            except Exception:
                pass
        return self.scrcpy.stop_headless_feed()

    def get_latest_frame(self) -> bytes | None:
        """Returns the latest headless-feed frame (or None if unavailable)."""
        try:
            return self.scrcpy.get_latest_frame()
        except Exception:
            return None

    # --- Sub-100ms live stream ---

    def start_live_stream(
        self,
        max_size: int = 960,
        bit_rate: str | int = "2M",
        max_fps: float | None = 30,
        new_display: str | None = None,
        timeout: float = 20.0,
    ) -> bool:
        """Starts a direct H.264 live stream and routes screenshots through it."""
        ok = self.scrcpy.start_live_stream(
            max_size=max_size,
            bit_rate=bit_rate,
            max_fps=max_fps,
            new_display=new_display,
            timeout=timeout,
        )
        if ok and hasattr(self.bridge, "set_frame_provider"):
            try:
                self.bridge.set_frame_provider(self.scrcpy.get_latest_frame)
            except Exception:
                pass
        return ok

    def stop_live_stream(self) -> bool:
        """Stops the live stream and restores plain screencap captures."""
        if hasattr(self.bridge, "clear_frame_provider"):
            try:
                # Only clear if no other feed is still providing frames.
                if not self.scrcpy.is_feed_running():
                    self.bridge.clear_frame_provider()
            except Exception:
                pass
        return self.scrcpy.stop_live_stream()

    def is_live_running(self) -> bool:
        """Checks if the sub-100ms live stream is running."""
        return self.scrcpy.is_live_running()

    def get_live_stats(self) -> dict:
        """Returns live-stream latency measurements (age_ms, fps, ...)."""
        return self.scrcpy.get_live_stats()

    def unlock_device(self, pin: str | None = None) -> None:
        """Unlocks device screen with PIN."""
        self.scrcpy.unlock_device(pin=pin)

    # --- Lifecycle ---


__all__ = [
    "AndroidController",
]
