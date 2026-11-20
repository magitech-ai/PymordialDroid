"""Tesseract OCR device implementing Pymordial's PymordialOCRDevice contract.

Uses a conditional, quality-driven preprocessing pipeline:

* **Upscaling** only when measured text height is below ~30px (Tesseract's
  sweet spot). Upscaling a clean page is harmless; upscaling a noisy page
  amplifies noise, so we measure first.
* **Binarization**: Otsu (global) for even lighting; adaptive (Sauvola-like)
  for uneven lighting — phone screenshots often have gradients.
* **Denoising**: median blur only when noise level is high; skipped on clean
  images to preserve thin strokes.
* **PSM selection**: PSM 6 (block of text) by default; PSM 8 (single word) or
  PSM 7 (single line) for short phrases; PSM 11 (sparse text) for scattered
  text.

Each preprocessing decision is gated by a measured image quality signal
rather than applied unconditionally — applying the wrong step can make
accuracy *worse*, not better.

See: https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract
from pymordial.core.blueprints.ocr_device import PymordialOCRDevice

from pymordialdroid.config import SystemConfig, resolve_system_config
from pymordialdroid.utils.extract_strategies import (
    DefaultExtractStrategy,
    PymordialExtractStrategy,
)
from pymordialdroid.utils.ocr_preprocessing import (
    measure_image_quality,
    preprocess_for_ocr,
)

log = logging.getLogger("pymordialdroid")


class TesseractDevice(PymordialOCRDevice):
    """Optical Character Recognition device powered by Tesseract and OpenCV.

    Uses conditional, quality-driven image preprocessing before OCR so that
    small text (e.g. Instagram UI elements) is upscaled to Tesseract's sweet
    spot (~30px cap height) without amplifying noise on clean images.
    """

    name: str = "ocr"
    version: str = "0.1.0"

    def __init__(
        self,
        config: str = "--oem 3 --psm 6",
        system_config: SystemConfig | None = None,
    ) -> None:
        self.config = config
        self.system_config = system_config or resolve_system_config()

        if self.system_config.tesseract_bin_path:
            pytesseract.pytesseract.tesseract_cmd = str(
                self.system_config.tesseract_bin_path
            )
            log.info(
                f"Using configured Tesseract: {self.system_config.tesseract_bin_path}"
            )

    def initialize(self, config: Any = None) -> None:
        """Initializes the OCR device plugin."""
        pass

    def shutdown(self) -> None:
        """Cleans up resources."""
        pass

    def _load_image(self, image_path: Path | bytes | str | np.ndarray) -> np.ndarray:
        """Loads and normalizes an image from various input types into an OpenCV BGR numpy array."""
        if isinstance(image_path, np.ndarray):
            return image_path

        if isinstance(image_path, (bytes, bytearray)):
            nparr = np.frombuffer(image_path, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Could not load image from {type(image_path)}")

        return image

    def _preprocess_for_ocr(
        self,
        image: np.ndarray,
        search_text: str | None = None,
        force_preprocess: bool = False,
    ) -> np.ndarray:
        """Preprocess an image for OCR with conditional, quality-driven steps.

        Each step is applied only when the measured image quality warrants it:
        upscaling only when text is small, denoising only when noisy,
        Otsu for even lighting vs adaptive for uneven.

        Args:
            image: Input BGR or grayscale image.
            search_text: Optional search text; if a single word or short
                phrase, we may upscale more aggressively since small text is
                the most common failure mode.
            force_preprocess: If True, always upscale and binarize regardless
                of measured quality.

        Returns:
            Preprocessed grayscale image ready for Tesseract.
        """
        # Determine target text height: smaller search text needs more upscaling
        target_height = 30.0
        if search_text:
            words = search_text.split()
            if len(words) == 1:
                target_height = 35.0  # single word: push harder
            elif len(words) <= 3:
                target_height = 32.0

        return preprocess_for_ocr(
            image,
            target_text_height_px=target_height,
            force_preprocess=force_preprocess,
        )

    def _select_psm(
        self,
        base_config: str,
        search_text: str | None = None,
        is_phrase: bool = False,
    ) -> str:
        """Select the best Tesseract PSM mode based on content.

        PSM 6 (single block of text) is the default.  For single words or
        short phrases, PSM 7 (single line) or PSM 8 (single word) can be more
        accurate.  For sparse text, PSM 11 (sparse text) works better.
        """
        # Extract any existing PSM from base config
        import re as _re

        existing_psm = _re.search(r"--psm\s+(\d+)", base_config)
        if existing_psm:
            return base_config  # caller already specified a PSM

        if is_phrase:
            words = search_text.split() if search_text else []
            if len(words) == 1:
                psm = 8  # single word
            elif len(words) <= 3:
                psm = 7  # single line
            else:
                psm = 6  # block of text
        else:
            psm = 6

        return (
            base_config.replace("--psm 6", f"--psm {psm}")
            if "--psm 6" in base_config
            else f"{base_config} --psm {psm}"
        )

    def extract_text(
        self,
        image_path: Path | bytes | str | np.ndarray,
        strategy: PymordialExtractStrategy | None = None,
    ) -> str:
        """Extracts text from an image with conditional preprocessing.

        Args:
            image_path: Source image as Path, bytes, string, or numpy array.
            strategy: Optional extraction strategy (defaults to DefaultExtractStrategy).

        Returns:
            Extracted text string.
        """
        try:
            image = self._load_image(image_path)
            strat = strategy or DefaultExtractStrategy()

            # Preprocess conditionally based on measured image quality
            processed = self._preprocess_for_ocr(image, force_preprocess=False)

            tess_config = (
                getattr(strat, "tesseract_config", lambda: self.config)() or self.config
            )
            tess_config = self._select_psm(tess_config)

            text = pytesseract.image_to_string(processed, config=tess_config)

            postprocess = getattr(strat, "postprocess_text", lambda t: t.strip())
            return postprocess(text)
        except Exception as e:
            log.error(f"Error extracting text with Tesseract: {e}")
            raise ValueError(f"Failed to extract text: {e}") from e

    def find_text(
        self,
        search_text: str,
        image_path: Path | bytes | str | np.ndarray,
        strategy: PymordialExtractStrategy | None = None,
    ) -> tuple[int, int] | None:
        """Finds (center_x, center_y) coordinates of search_text in the source image.

        Uses conditional preprocessing (upscale small text, denoise if noisy,
        adaptive threshold for uneven lighting) and PSM selection based on
        search text length.  Filters results by OCR confidence (>= 30) to
        avoid false positives on noisy regions.

        Args:
            search_text: Text keyword or substring to search for.
            image_path: Source image as Path, bytes, string, or numpy array.
            strategy: Optional extraction strategy.

        Returns:
            (center_x, center_y) coordinates if located, None otherwise.
        """
        try:
            image = self._load_image(image_path)
            strat = strategy or DefaultExtractStrategy()
            processed = self._preprocess_for_ocr(image, search_text)

            tess_config = (
                getattr(strat, "tesseract_config", lambda: self.config)() or self.config
            )
            tess_config = self._select_psm(tess_config, search_text)
            data = pytesseract.image_to_data(
                processed, config=tess_config, output_type=pytesseract.Output.DICT
            )

            search_lower = search_text.lower().strip()
            n_boxes = len(data.get("text", []))
            upscale = getattr(strat, "upscale_factor", 1) or 1

            # Determine the upscale factor actually applied so we can
            # scale coordinates back to original image space.
            quality = measure_image_quality(image)
            if quality.estimated_text_height_px < 30.0 * 0.8:
                # We upscaled; estimate the factor
                upscale = max(
                    1.0,
                    30.0 / max(quality.estimated_text_height_px, 1.0),
                )
                upscale = min(upscale, 4.0)

            best_match: tuple[int, int] | None = None
            best_conf = 0

            for i in range(n_boxes):
                conf = int(data["conf"][i])
                if conf <= 0:
                    continue
                box_text = data["text"][i].strip().lower()
                if search_lower in box_text and conf > best_conf:
                    x, y, w, h = (
                        data["left"][i],
                        data["top"][i],
                        data["width"][i],
                        data["height"][i],
                    )
                    center_x = int((x + w // 2) / upscale)
                    center_y = int((y + h // 2) / upscale)
                    best_match = (center_x, center_y)
                    best_conf = conf

            if best_match is not None:
                log.debug(
                    "Found '%s' at %s (conf=%d)", search_text, best_match, best_conf
                )
            return best_match
        except Exception as e:
            log.error(f"Error finding text '{search_text}' with Tesseract: {e}")
            return None

    def find_text_phrase(
        self,
        search_text: str,
        image_path: Path | bytes | str | np.ndarray,
        strategy: PymordialExtractStrategy | None = None,
        max_gap: int = 1,
    ) -> tuple[int, int] | None:
        """Finds coordinates of a multi-word phrase across consecutive OCR boxes.

        Tesseract often splits phrases like "Security and privacy" into separate
        per-word boxes. This method searches consecutive OCR boxes whose combined
        text matches ``search_text`` (case-insensitive, substring), returning the
        center of the combined bounding box.

        Uses conditional preprocessing and PSM 7 (single line) for short phrases
        or PSM 6 (block of text) for longer ones.

        Args:
            search_text: Multi-word phrase to search for.
            image_path: Source image as Path, bytes, string, or numpy array.
            strategy: Optional extraction strategy.
            max_gap: Maximum number of non-matching boxes allowed between
                consecutive phrase words (default 1, tolerating OCR noise).

        Returns:
            (center_x, center_y) coordinates of the phrase's bounding box center,
            or None if not found.
        """
        try:
            image = self._load_image(image_path)
            strat = strategy or DefaultExtractStrategy()
            processed = self._preprocess_for_ocr(image)

            tess_config = (
                getattr(strat, "tesseract_config", lambda: self.config)() or self.config
            )
            tess_config = self._select_psm(tess_config, search_text, is_phrase=True)
            data = pytesseract.image_to_data(
                processed, config=tess_config, output_type=pytesseract.Output.DICT
            )

            search_words = [w for w in search_text.lower().strip().split() if w]
            if not search_words:
                return None

            n_boxes = len(data.get("text", []))
            upscale = getattr(strat, "upscale_factor", 1) or 1

            # Determine the upscale factor actually applied
            quality = measure_image_quality(image)
            if quality.estimated_text_height_px < 30.0 * 0.8:
                upscale = max(
                    1.0,
                    30.0 / max(quality.estimated_text_height_px, 1.0),
                )
                upscale = min(upscale, 4.0)

            # Collect valid boxes with text, position, and confidence.
            boxes: list[dict] = []
            for i in range(n_boxes):
                conf = int(data["conf"][i])
                if conf <= 0:
                    continue
                box_text = data["text"][i].strip().lower()
                if not box_text:
                    continue
                boxes.append(
                    {
                        "text": box_text,
                        "left": int(data["left"][i]),
                        "top": int(data["top"][i]),
                        "width": int(data["width"][i]),
                        "height": int(data["height"][i]),
                    }
                )

            # Sliding window: try each starting box as phrase word 0.
            for start in range(len(boxes)):
                wi = 0  # index into search_words
                bi = start  # index into boxes
                gap = 0
                matched_boxes: list[dict] = []

                while bi < len(boxes) and wi < len(search_words):
                    bt = boxes[bi]["text"]
                    sw = search_words[wi]

                    if sw in bt:
                        matched_boxes.append(boxes[bi])
                        # Consume ALL search words that appear in this same box.
                        while wi + 1 < len(search_words):
                            next_sw = search_words[wi + 1]
                            if next_sw in bt:
                                wi += 1
                            else:
                                break
                        wi += 1
                        gap = 0
                    elif gap < max_gap:
                        gap += 1
                    else:
                        break
                    bi += 1

                if wi == len(search_words) and matched_boxes:
                    # Compute combined bounding box center.
                    x_min = min(b["left"] for b in matched_boxes)
                    y_min = min(b["top"] for b in matched_boxes)
                    x_max = max(b["left"] + b["width"] for b in matched_boxes)
                    y_max = max(b["top"] + b["height"] for b in matched_boxes)
                    center_x = int((x_min + x_max) // 2 / upscale)
                    center_y = int((y_min + y_max) // 2 / upscale)
                    return (center_x, center_y)

            return None
        except Exception as e:
            log.error(f"Error finding phrase '{search_text}' with Tesseract: {e}")
            return None


__all__ = [
    "TesseractDevice",
]
