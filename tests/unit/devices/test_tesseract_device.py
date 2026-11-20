"""Unit tests for TesseractDevice in pymordialdroid.devices."""

from unittest.mock import patch

import numpy as np
import pytest
from pymordial.core.blueprints.ocr_device import PymordialOCRDevice

from pymordialdroid.devices.tesseract_device import TesseractDevice
from pymordialdroid.utils.extract_strategies import DefaultExtractStrategy
from pymordialdroid.utils.ocr_preprocessing import ImageQuality


@pytest.fixture
def mock_image_quality(mocker):
    """Mock measure_image_quality to return a quality with sufficient text height
    so no upscaling happens during tests."""
    mocker.patch(
        "pymordialdroid.devices.tesseract_device.measure_image_quality",
        return_value=ImageQuality(
            width=100,
            height=100,
            mean_brightness=128.0,
            brightness_std=40.0,
            estimated_text_height_px=30.0,
            noise_level=5.0,
            is_dark_background=False,
        ),
    )


def test_tesseract_device_satisfies_contract():
    """TesseractDevice must be a subclass of PymordialOCRDevice."""
    assert issubclass(TesseractDevice, PymordialOCRDevice)
    ocr = TesseractDevice()
    assert ocr.name == "ocr"
    assert ocr.version == "0.1.0"


def test_tesseract_device_init():
    """Test device initialization."""
    device = TesseractDevice()
    assert device.name == "ocr"
    assert device.version == "0.1.0"


def test_tesseract_device_load_image():
    """Test image loading from numpy array, bytes, and invalid input."""
    device = TesseractDevice()

    # 1. Numpy array
    arr = np.zeros((50, 50, 3), dtype=np.uint8)
    loaded = device._load_image(arr)
    assert np.array_equal(loaded, arr)

    # 2. Invalid path
    with pytest.raises(ValueError, match="Could not load image"):
        device._load_image("non_existent_image_12345.png")


def test_extract_text_mocked():
    """Test text extraction with mocked pytesseract."""
    device = TesseractDevice()
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch("pytesseract.image_to_string", return_value="  Detected Text  \n"):
        text = device.extract_text(img)
        assert text == "Detected Text"


def test_find_text_mocked(mock_image_quality):
    """Test finding text bounding box and center coordinate calculation."""
    device = TesseractDevice()
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    mock_dict = {
        "text": ["", "Hello", "World"],
        "conf": [0, 95, 90],
        "left": [0, 20, 60],
        "top": [0, 30, 30],
        "width": [0, 20, 30],
        "height": [0, 10, 10],
    }

    # Custom strategy with upscale_factor=1 for simplicity
    strategy = DefaultExtractStrategy(upscale_factor=1)

    with patch("pytesseract.image_to_data", return_value=mock_dict):
        coords = device.find_text("world", img, strategy=strategy)
        # Center of (60, 30, 30, 10) is (60 + 15, 30 + 5) = (75, 35)
        assert coords == (75, 35)

        # Test text not found
        coords_not_found = device.find_text("NonExistent", img, strategy=strategy)
        assert coords_not_found is None


def test_find_text_phrase_single_box(mock_image_quality):
    """find_text_phrase matches a phrase within a single OCR box."""
    device = TesseractDevice()
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    mock_dict = {
        "text": ["", "Security and privacy", ""],
        "conf": [0, 95, 0],
        "left": [0, 20, 80],
        "top": [0, 30, 30],
        "width": [0, 60, 10],
        "height": [0, 10, 10],
    }

    strategy = DefaultExtractStrategy(upscale_factor=1)

    with patch("pytesseract.image_to_data", return_value=mock_dict):
        coords = device.find_text_phrase("Security and privacy", img, strategy=strategy)
        # Center of (20, 30, 60, 10) is (20 + 30, 30 + 5) = (50, 35)
        assert coords == (50, 35)


def test_find_text_phrase_across_boxes(mock_image_quality):
    """find_text_phrase matches a phrase split across consecutive OCR boxes."""
    device = TesseractDevice()
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    # "Security" then "and" then "privacy" in separate boxes
    mock_dict = {
        "text": ["", "Security", "and", "privacy", ""],
        "conf": [0, 90, 85, 88, 0],
        "left": [0, 20, 85, 110, 200],
        "top": [0, 30, 30, 30, 30],
        "width": [0, 60, 25, 40, 10],
        "height": [0, 10, 10, 10, 10],
    }

    strategy = DefaultExtractStrategy(upscale_factor=1)

    with patch("pytesseract.image_to_data", return_value=mock_dict):
        coords = device.find_text_phrase("Security and privacy", img, strategy=strategy)
        # Combined box: left=20, top=30, right=150, bottom=40
        # Center = ((20+150)//2, (30+40)//2) = (85, 35)
        assert coords == (85, 35)


def test_find_text_phrase_with_gap(mock_image_quality):
    """find_text_phrase tolerates a non-matching box between phrase words."""
    device = TesseractDevice()
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    # "Security" [noise] "and" "privacy" - one non-matching box between
    mock_dict = {
        "text": ["", "Security", "noise", "and", "privacy", ""],
        "conf": [0, 90, 50, 85, 88, 0],
        "left": [0, 20, 60, 100, 130, 200],
        "top": [0, 30, 30, 30, 30, 30],
        "width": [0, 60, 30, 25, 40, 10],
        "height": [0, 10, 10, 10, 10, 10],
    }

    strategy = DefaultExtractStrategy(upscale_factor=1)

    with patch("pytesseract.image_to_data", return_value=mock_dict):
        # max_gap=1 allows skipping "noise"
        coords = device.find_text_phrase(
            "Security and privacy", img, strategy=strategy, max_gap=1
        )
        # Combined: left=20, top=30, right=170, bottom=40
        # Center = ((20+170)//2, (30+40)//2) = (95, 35)
        assert coords == (95, 35)


def test_find_text_phrase_not_found(mock_image_quality):
    """find_text_phrase returns None when phrase is not present."""
    device = TesseractDevice()
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    mock_dict = {
        "text": ["", "Hello", "World"],
        "conf": [0, 95, 90],
        "left": [0, 20, 60],
        "top": [0, 30, 30],
        "width": [0, 20, 30],
        "height": [0, 10, 10, 10],
    }

    strategy = DefaultExtractStrategy(upscale_factor=1)

    with patch("pytesseract.image_to_data", return_value=mock_dict):
        coords = device.find_text_phrase("NonExistent Phrase", img, strategy=strategy)
        assert coords is None


def test_find_text_phrase_max_gap_zero(mock_image_quality):
    """find_text_phrase with max_gap=0 requires adjacent boxes."""
    device = TesseractDevice()
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    mock_dict = {
        "text": ["", "Security", "noise", "and", "privacy", ""],
        "conf": [0, 90, 50, 85, 88, 0],
        "left": [0, 20, 60, 100, 130, 200],
        "top": [0, 30, 30, 30, 30, 30],
        "width": [0, 60, 30, 25, 40, 10],
        "height": [0, 10, 10, 10, 10, 10],
    }

    strategy = DefaultExtractStrategy(upscale_factor=1)

    with patch("pytesseract.image_to_data", return_value=mock_dict):
        # max_gap=0 means "noise" breaks the match
        coords = device.find_text_phrase(
            "Security and privacy", img, strategy=strategy, max_gap=0
        )
        assert coords is None
