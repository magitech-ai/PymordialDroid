"""OpenCV-based image preprocessing for improved OCR accuracy.

Provides per-image quality measurement and conditional preprocessing:
Tesseract works best on 300 DPI-equivalent images with ~30px cap height.
Small text needs upscaling; uneven lighting needs adaptive thresholding;
noisy images need denoising.  Crucially, preprocessing is *conditional*:
applying the wrong step (e.g. upscaling a noisy page, or Otsu on uneven
lighting) can make accuracy *worse*, so each step is gated by a measured
quality signal rather than applied unconditionally.

See: https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("pymordialdroid")


@dataclass(frozen=True)
class ImageQuality:
    """Measured characteristics of an image that drive preprocessing choices."""

    width: int
    height: int
    mean_brightness: float
    brightness_std: float
    estimated_text_height_px: float
    noise_level: float
    is_dark_background: bool


def measure_image_quality(image: np.ndarray) -> ImageQuality:
    """Measure image characteristics to guide preprocessing decisions.

    Returns an ImageQuality object with fields that each drive a specific
    preprocessing decision:

    * ``estimated_text_height_px`` drives upscaling (Tesseract wants ~30px).
    * ``brightness_std`` drives adaptive vs global thresholding (uneven
      lighting needs adaptive; even lighting can use Otsu).
    * ``noise_level`` drives denoising (skip denoising on clean images,
      since it can destroy thin strokes).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    h, w = gray.shape
    mean_brightness = float(gray.mean())
    brightness_std = float(gray.std())

    # Estimate text height from the distribution of dark pixel runs in
    # column projections.  Dark vertical bands correspond to text lines;
    # the median run height approximates cap height.
    col_means = gray.mean(axis=0)
    dark_cols = col_means < mean_brightness * 0.6
    runs = []
    run_start = None
    for i, is_dark in enumerate(dark_cols):
        if is_dark and run_start is None:
            run_start = i
        elif not is_dark and run_start is not None:
            runs.append(run_start)
            run_start = None
    if run_start is not None:
        runs.append(run_start)
    heights = [b - a for a, b in zip(runs[:-1], runs[1:])] if len(runs) > 1 else []
    estimated_text_height = float(np.median(heights)) if heights else float(h) / 20.0

    # Noise estimate: high-frequency energy after light blur.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = float(np.std(gray.astype(np.float32) - blurred.astype(np.float32)))

    return ImageQuality(
        width=w,
        height=h,
        mean_brightness=mean_brightness,
        brightness_std=brightness_std,
        estimated_text_height_px=estimated_text_height,
        noise_level=noise,
        is_dark_background=mean_brightness < 128,
    )


def upscale_image(
    image: np.ndarray, target_text_height_px: float = 30.0
) -> tuple[np.ndarray, float]:
    """Upscale image so estimated text height reaches target_text_height_px.

    Returns the upscaled image and the scale factor applied.
    """
    # We need an estimate of current text height; the caller should have
    # measured it.  If we don't have a measurement, assume a conservative
    # default and upscale modestly.
    scale = 2.0  # default 2x upscale
    if target_text_height_px > 0:
        # Assume current text height is roughly 1/20 of image height
        # (typical for UI screenshots); scale to bring it to target.
        current_est = image.shape[0] / 20.0
        if current_est > 0:
            scale = target_text_height_px / current_est
    scale = max(1.0, min(scale, 4.0))  # clamp between 1x and 4x
    if scale > 1.01:
        new_w = int(image.shape[1] * scale)
        new_h = int(image.shape[0] * scale)
        upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        upscaled = image
    return upscaled, scale


def denoise_image(image: np.ndarray, noise_level: float) -> np.ndarray:
    """Denoise image if noise level warrants it.

    Median blur preserves thin stroke edges better than Gaussian blur,
    which matters for small text.  Skipped entirely when noise is low
    because denoising can destroy fine strokes.
    """
    if noise_level < 8.0:
        return image  # clean enough; denoising would hurt more than help
    if noise_level < 20.0:
        return cv2.medianBlur(image, 3)
    return cv2.medianBlur(image, 5)


def binarize_image(image: np.ndarray, brightness_std: float) -> np.ndarray:
    """Binarize image using Otsu (even lighting) or adaptive thresholding
    (uneven lighting).

    Otsu is a single global threshold; it fails when brightness varies
    across the page.  Adaptive thresholding computes a local threshold
    per pixel neighborhood, which handles uneven lighting but can
    amplify noise on clean images.
    """
    if brightness_std < 40.0:
        # Even lighting: global Otsu is fine and avoids adaptive artifacts
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Uneven lighting: adaptive (Sauvola-like via Gaussian)
        binary = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            10,
        )
    return binary


def preprocess_for_ocr(
    image: np.ndarray,
    target_text_height_px: float = 30.0,
    force_preprocess: bool = False,
) -> np.ndarray:
    """Preprocess an image for OCR with conditional, quality-driven steps.

    Each step is applied only when the measured image quality warrants it:

    * Upscaling: only when estimated text height is below target.
    * Denoising: only when noise level is above threshold.
    * Binarization: Otsu for even lighting, adaptive for uneven.

    Args:
        image: Input BGR or grayscale image.
        target_text_height_px: Target cap height in pixels (Tesseract wants ~30).
        force_preprocess: If True, always upscale and binarize regardless of
            measured quality (useful when you know the image is low quality).

    Returns:
        Preprocessed grayscale image ready for Tesseract.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    quality = measure_image_quality(gray)

    log.debug(
        "Image quality: w=%d h=%d brightness=%.0f std=%.0f "
        "text_h=%.1f noise=%.1f dark_bg=%s",
        quality.width,
        quality.height,
        quality.mean_brightness,
        quality.brightness_std,
        quality.estimated_text_height_px,
        quality.noise_level,
        quality.is_dark_background,
    )

    # Upscale only if text is small
    if (
        force_preprocess
        or quality.estimated_text_height_px < target_text_height_px * 0.8
    ):
        gray, scale = upscale_image(gray, target_text_height_px)
        log.debug("Upscaled by %.2fx to reach target text height", scale)
        # Re-measure after upscale
        quality = measure_image_quality(gray)

    # Denoise only if noisy
    gray = denoise_image(gray, quality.noise_level)

    # Binarize based on lighting evenness
    gray = binarize_image(gray, quality.brightness_std)

    return gray


def preprocess_for_phrase_ocr(
    image: np.ndarray,
    target_text_height_px: float = 30.0,
) -> np.ndarray:
    """Preprocess for phrase-level OCR (find_text_phrase).

    Phrase matching benefits from slightly more aggressive upscaling since
    multi-word boxes need higher resolution to separate individual words.
    """
    return preprocess_for_ocr(image, target_text_height_px=target_text_height_px)


def get_optimal_psm(image: np.ndarray, search_text: str | None = None) -> str:
    """Select the best Tesseract PSM mode based on image content.

    PSM 6 (single block of text) is the default.  For single words or short
    phrases, PSM 7 (single line) or PSM 8 (single word) can be more accurate.
    For sparse text, PSM 11 (sparse text) works better.
    """
    if search_text:
        words = search_text.split()
        if len(words) == 1:
            return "--psm 8"  # single word
        if len(words) <= 3:
            return "--psm 7"  # single line
    # Default: single block of text
    return "--psm 6"


__all__ = [
    "ImageQuality",
    "measure_image_quality",
    "upscale_image",
    "denoise_image",
    "binarize_image",
    "preprocess_for_ocr",
    "preprocess_for_phrase_ocr",
    "get_optimal_psm",
]
