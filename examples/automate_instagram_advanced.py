"""Advanced Instagram automation example.

Demonstrates end-to-end device automation on a real Android device:

1. Opening Instagram and dismissing permission dialogs
2. Scrolling through the feed with swipe_to_find
3. Double-tapping to like posts
4. Finding specific content by OCR text
5. Taking screenshots at each step

Prerequisites:
- Instagram app installed (com.instagram.android)
- Device connected via ADB (USB or WiFi)
- Tesseract OCR installed

Run with:
    uv run python examples/automate_instagram_advanced.py
"""

import time
from pathlib import Path

from pymordialdroid.android_controller import AndroidController
from pymordialdroid.config import SystemConfig

ADB_PATH = Path("/home/vanluther/.gemini/antigravity/scratch/revomon/platform-tools/adb")
SCRCPY_PATH = Path("/tmp/dummy_scrcpy")


def dismiss_dialogs(controller: AndroidController, max_attempts: int = 3) -> None:
    """Dismiss common Instagram dialogs (notifications, permissions, etc.)."""
    for attempt in range(max_attempts):
        screenshot = controller.adb.capture_screenshot()
        if screenshot is None:
            break

        # Look for common dialog buttons
        for phrase in ["Allow", "Don't allow", "Not Now", "Skip", "Later", "OK"]:
            coords = controller.ui.find_text_phrase(phrase, screenshot)
            if coords:
                print(f"    Found '{phrase}' at {coords} - tapping")
                controller.adb.tap(coords)
                time.sleep(2)
                screenshot = controller.adb.capture_screenshot()
                if screenshot is None:
                    break
                break
        else:
            # No dialog buttons found on this screen
            break


def scroll_feed(controller: AndroidController, times: int = 3) -> None:
    """Scroll the Instagram feed up multiple times."""
    for i in range(times):
        print(f"    Scroll {i + 1}/{times}...")
        controller.adb.swipe(500, 1600, 500, 400, 800)
        time.sleep(2.5)


def find_and_like_post(controller: AndroidController) -> bool:
    """Find a post and double-tap to like it.

    Uses swipe_to_find to scroll until a likeable element appears,
    then double-taps it (Instagram's like gesture).
    """
    from pymordial.ui.text import PymordialText

    # Try to find "Like" text on screen
    element = PymordialText(
        label="Like", element_text="Like", og_resolution=(1080, 2340)
    )
    found = controller.find_and_tap_with_retry(
        element, times=2, max_swipes=5, interval=2.0
    )
    return found


def main() -> None:
    """Run the advanced Instagram automation demo."""
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
    print("PymordialDroid - Advanced Instagram Automation")
    print("=" * 60)

    try:
        # Step 1: Open Instagram
        print("\n[1/6] Opening Instagram...")
        controller.open_app("Instagram", "com.instagram.android")
        time.sleep(5)

        # Step 2: Dismiss dialogs
        print("\n[2/6] Dismissing dialogs...")
        dismiss_dialogs(controller)

        # Step 3: Read initial screen
        print("\n[3/6] Reading screen via OCR...")
        screenshot = controller.adb.capture_screenshot()
        Path("/tmp/instagram_advanced_1.png").write_bytes(screenshot)
        texts = controller.read_text(screenshot)
        print("    Screen text:")
        for t in texts[:15]:
            print(f"      - {t}")

        # Step 4: Scroll and like posts
        print("\n[4/6] Scrolling feed and liking posts...")
        posts_liked = 0
        for scroll_round in range(3):
            print(f"    Scroll round {scroll_round + 1}/3...")
            scroll_feed(controller, times=2)
            screenshot = controller.adb.capture_screenshot()
            Path(
                f"/tmp/instagram_advanced_scroll_{scroll_round}.png"
            ).write_bytes(screenshot)

            # Try to like a post
            if find_and_like_post(controller):
                posts_liked += 1
                print(f"    ✓ Liked post #{posts_liked}")
            else:
                print("    No likeable post found in this scroll")

        # Step 5: Final read
        print("\n[5/6] Final screen read...")
        screenshot = controller.adb.capture_screenshot()
        Path("/tmp/instagram_advanced_final.png").write_bytes(screenshot)
        texts = controller.read_text(screenshot)
        print("    Final screen text:")
        for t in texts[:15]:
            print(f"      - {t}")

        # Step 6: Go home
        print("\n[6/6] Going home...")
        controller.go_home()
        time.sleep(1)

        print("\n" + "=" * 60)
        print(f"Demo complete! Liked {posts_liked} post(s).")
        print("=" * 60)

    finally:
        controller.shutdown()
        print("Controller shut down.")


if __name__ == "__main__":
    main()
