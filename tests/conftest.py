"""
Test Fixtures and Configuration for Pixel Lab Test Suite

This file is automatically discovered by pytest and provides reusable test fixtures.
Fixtures are functions that create test data, avoiding duplication across tests.

Design decisions:
1. Use pytest's tmp_path fixture for temporary files (auto-cleanup)
2. Fixed random seed (42) for reproducible test results
3. Standard image sizes (100x100 for speed, 256x256 for visual patterns)
4. Create images once per test function (fast enough, keeps tests isolated)
"""

# pytest: Testing framework - provides fixtures, assertions, test discovery
# pathlib.Path: Modern path handling - works cross-platform
from pathlib import Path

# typing: Type hints for function signatures
from typing import Tuple

# numpy: Generate deterministic test data with fixed random seed
import numpy as np
import pytest

# Import our classes to create test fixtures
from pixel_lab import ImageGenerator

# ============================================================================
# Path Fixtures
# ============================================================================


@pytest.fixture
def temp_image_path(tmp_path: Path) -> Path:
    """
    Provides a temporary file path for test images.

    Args:
        tmp_path: pytest built-in fixture providing temporary directory

    Returns:
        Path object to non-existent test_image.png

    Design decision: Use tmp_path (pytest built-in) instead of tempfile module
    because pytest automatically cleans up after tests, preventing disk clutter.
    """
    return tmp_path / "test_image.png"


# ============================================================================
# Image Generator Fixtures
# ============================================================================


@pytest.fixture
def uniform_random_image(temp_image_path: Path) -> Path:
    """
    Generate a uniform random image for testing randomness metrics.

    This image should pass all randomness tests:
    - High entropy (≈ 8.0)
    - Chi-square p-value > 0.05
    - Low correlation (≈ 0.0)
    - Low MAD
    - Monte Carlo π error < 5%

    Design decisions:
    1. Size 100x100 = 30,000 bytes (fast to generate, large enough for statistics)
    2. Fixed seed (42) ensures reproducible test results across runs
    3. Use RandomState instead of global random for isolation

    Returns:
        Path to saved random image
    """
    gen = ImageGenerator(100, 100)

    # Use fixed seed for reproducibility
    # Design note: RandomState is isolated - doesn't affect other tests
    rng = np.random.RandomState(42)

    # Fill with uniform random bytes
    for i in range(gen.byte_count):
        gen.set_byte(i, rng.randint(0, 256))

    gen.save(str(temp_image_path))
    return temp_image_path


@pytest.fixture
def solid_color_image(temp_image_path: Path) -> Path:
    """
    Generate a solid gray image for testing non-random metrics.

    This image should show minimal randomness:
    - Low entropy (≈ 0.0)
    - Chi-square p-value ≈ 0.0 (fails uniformity)
    - Correlation undefined or 0.0 (all values same)
    - High MAD percentage

    Design decision: Use middle gray (128, 128, 128) to avoid edge effects
    of pure black/white which might trigger special cases.

    Returns:
        Path to saved solid color image
    """
    gen = ImageGenerator(100, 100)
    gen.fill(128, 128, 128)  # Middle gray
    gen.save(str(temp_image_path))
    return temp_image_path


@pytest.fixture
def gradient_image(temp_image_path: Path) -> Path:
    """
    Generate a gradient image for testing correlation metrics.

    This image should show high spatial correlation:
    - Ideal entropy (8.0) for RED and GREEN, zero for BLUE and moderate (≈ 7.0) for ALL.
    - High horizontal correlation (nearly 1.0) for READ and GREEN, zero for BLUE and low (≈ 0.25) for ALL
    - High vertical correlation (nearly 1.0) for READ and GREEN, zero for BLUE and low (≈ 0.25) for ALL
    - High diagonal correlation (nearly 1.0) for READ and GREEN, zero for BLUE and low (≈ 0.25) for ALL

    Design decision: Use 256x256 to get full byte value range (0-255)
    in both dimensions, creating smooth gradients.

    Returns:
        Path to saved gradient image
    """
    gen = ImageGenerator(256, 256)

    # Create gradient: R increases with x, G increases with y, B = 0
    for y in range(gen.height):
        for x in range(gen.width):
            gen.set_pixel(x, y, x, y, 0)

    gen.save(str(temp_image_path))
    return temp_image_path


@pytest.fixture
def small_image(temp_image_path: Path) -> Path:
    """
    Generate minimal 1x1 pixel image for edge case testing.

    Design decision: Smallest valid image to test boundary conditions
    like insufficient data for correlation or Monte Carlo tests.

    Returns:
        Path to saved 1x1 image
    """
    gen = ImageGenerator(1, 1)
    gen.set_pixel(0, 0, 255, 128, 64)
    gen.save(str(temp_image_path))
    return temp_image_path


# ============================================================================
# Helper Functions (not fixtures, but shared utilities)
# ============================================================================


def assert_valid_image_array(
    array: np.ndarray, expected_shape: Tuple[int, int, int]
) -> None:
    """
    Assert that array is a valid RGB image with expected dimensions.

    Args:
        array: numpy array to validate
        expected_shape: tuple (height, width, 3)

    Design decision: Centralize image validation logic to avoid
    duplication across tests. All RGB images must be uint8 with 3 channels.
    """
    assert array.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {array.shape}"
    )
    assert array.dtype == np.uint8, f"Expected dtype uint8, got {array.dtype}"
    assert np.all(array >= 0) and np.all(array <= 255), (
        "RGB values must be in range [0, 255]"
    )


def assert_entropy_in_range(
    entropy: float, min_val: float, max_val: float, message: str = ""
) -> None:
    """
    Assert entropy is within expected range for a given image type.

    Args:
        entropy: computed entropy value
        min_val: minimum acceptable entropy
        max_val: maximum acceptable entropy
        message: optional context message

    Design decision: Entropy ranges are well-known for different image types,
    so we can use ranges instead of exact values (which vary slightly).
    """
    assert min_val <= entropy <= max_val, (
        f"{message} Entropy {entropy:.4f} outside range [{min_val}, {max_val}]"
    )


def assert_correlation_type(correlation: float, correlation_type: str) -> None:
    """
    Assert correlation matches expected type (one, high, low, or zero).

    Args:
        correlation: computed correlation coefficient
        correlation_type: 'one' (>0.9), 'high' (>0.5), 'low' (<0.3), or 'zero' (<0.1)

    Design decision: Use qualitative categories rather than exact thresholds
    since correlation values can vary slightly with different images.
    """
    abs_corr = abs(correlation)

    if correlation_type == "one":
        assert abs_corr > 0.9, (
            f"Expected practically perfect correlation (>0.9), got {correlation:.4f}"
        )
    elif correlation_type == "high":
        assert abs_corr > 0.5, (
            f"Expected high correlation (>0.5), got {correlation:.4f}"
        )
    elif correlation_type == "low":
        assert abs_corr < 0.3, f"Expected low correlation (<0.3), got {correlation:.4f}"
    elif correlation_type == "zero":
        assert abs_corr < 0.1, (
            f"Expected negligible correlation (<0.1), got {correlation:.4f}"
        )
    else:
        raise ValueError(f"Unknown correlation type: {correlation_type}")
