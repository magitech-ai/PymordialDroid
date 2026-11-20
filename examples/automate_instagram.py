"""Instagram automation example using PymordialDroid.

This script demonstrates end-to-end device automation on a real Android device:
1. Opens Instagram
2. Handles the notification permission dialog
3. Reads screen text via OCR
4. Finds and taps UI elements
5. Waits for elements to appear
6. Captures screenshots

Prerequisites:
- Instagram app installed (com.instagram.android)
- Device connected via ADB (USB or WiFi)
- PyAV installed for live streaming (optional)

Run with:
    uv run python examples/automate_instagram.py
"""

from pathlib import Path
import time

from pymordialdroid.config import SystemConfig
from pymordialdroid.android_controller import AndroidController
from pymordial.ui.text import PymordialText

# Point at the ADB binary from the Android platform-tools
ADB_PATH = Path("/home/vanluther/.gemini/antigravity/scratch/revomon/platform-tools/adb")
SCRCPY_PATH = Path("/tmp/dummy_scrcpy")  # dummy, not needed for ADB-only ops


def main() -> None:
    """Run the Instagram automation demo."""
    cfg = SystemConfig(
        adb_bin_path=ADB_PATH,
        scrcpy_bin_path=SCRCPY_PATH,
        tesseract_bin_path=Path("/usr/bin/tesseract"),
    )

    controller = AndroidController(
        ip="172.20.8.50",
        port=5555,
        device_name="S24 Ultra",
        system_config=cfg,
    )

    print("=" * 60)
    print("PymordialDroid - Instagram Automation Demo")
    print("=" * 60)

    try:
        # Step 1: Open Instagram
        print("\n[1/5] Opening Instagram...")
        result = controller.open_app("Instagram", "com.instagram.android")
        print(f"    Open result: {result}")
        time.sleep(5)

        # Step 2: Read what's on screen via OCR
        print("\n[2/5] Reading screen via OCR...")
        screenshot = controller.adb.capture_screenshot()
        if screenshot:
            Path("/tmp/instagram_screenshot.png").write_bytes(screenshot)
            print(f"    Screenshot saved ({len(screenshot)} bytes)")

        texts = controller.read_text(screenshot)
        print("    Text found on screen:")
        for t in texts[:20]:
            print(f"      - {t}")

        # Step 3: Handle notification permission dialog if present
        print("\n[3/5] Handling notification dialog...")
        allow_element = PymordialText(
            label="Allow", element_text="Allow", og_resolution=(1080, 2340)
        )
        if controller.wait_for_element(allow_element, timeout=5.0, interval=1.0):
            print("    'Allow' button found - tapping it...")
            controller.find_and_tap(allow_element)
            time.sleep(2)
        else:
            print("    No notification dialog found (or already dismissed)")

        # Step 4: Find and tap on feed content
        print("\n[4/5] Finding and tapping feed content...")
        screenshot2 = controller.adb.capture_screenshot()
        for phrase in ["Reels", "Messages", "Profile", "Search", "Create"]:
            element = PymordialText(
                label=phrase, element_text=phrase, og_resolution=(1080, 2340)
            )
            coords = controller.ui.find_text_phrase(phrase, screenshot2)
            if coords:
                print(f"    Found '{phrase}' at {coords} - tapping...")
                controller.adb.tap(coords)
                time.sleep(2)
                break
        else:
            print("    No navigation element found, scrolling feed instead...")
            controller.adb.swipe(500, 1500, 500, 500, 500)
            time.sleep(2)

        # Step 5: Read screen after interaction
        print("\n[5/5] Reading screen after interaction...")
        screenshot3 = controller.adb.capture_screenshot()
        texts3 = controller.read_text(screenshot3)
        print("    Text after interaction:")
        for t in texts3[:15]:
            print(f"      - {t}")

        print("\n" + "=" * 60)
        print("Demo complete!")
        print("=" * 60)

    finally:
        controller.shutdown()
        print("Controller shut down.")


if __name__ == "__main__":
    main()
