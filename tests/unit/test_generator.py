"""
Unit Tests for ImageGenerator Class

Tests organized by functionality:
1. Constructor and properties (3 tests)
2. Direct pixel assignment (3 tests)
3. Direct byte assignment (3 tests)
4. Recursive operations (2 tests)
5. File I/O (1 test)
6. Edge cases - Week 2 (10 tests)

Total: 22 tests

Design philosophy:
- Test one thing per test function
- Use descriptive test names (test_[what]_[condition]_[expected])
- Arrange-Act-Assert pattern
- Test both success and failure cases
- Explicit type hints for all parameters and returns
"""

# pytest: Testing framework - provides assertions and test discovery
# os.path: Check file existence after save
import os

# pathlib.Path: Type hints for path parameters
from pathlib import Path

# typing: Type hints for function signatures
from typing import Tuple

# numpy: Validate array contents and shapes
import numpy as np
import pytest

# Import test utilities from conftest
from conftest import assert_valid_image_array

# PIL.Image: Verify saved images are valid
from PIL import Image

# Import classes under test
from pixel_lab import ImageGenerator

# ============================================================================
# 1. Constructor and Properties Tests
# ============================================================================


def test_valid_image_array() -> None:
    """
    Assert that array is a valid RGB image with expected dimensions.
    """
    # Test small image
    gen = ImageGenerator(10, 20)
    assert_valid_image_array(gen.get_numpy_array(), (gen.height, gen.width, 3))


def test_constructor_valid_dimensions() -> None:
    """
    Test that constructor creates correct array shape with valid dimensions.

    Design decision: Test multiple sizes to ensure no hardcoded values.
    """
    # Test small image
    gen = ImageGenerator(10, 20)
    assert gen.width == 10
    assert gen.height == 20
    assert gen.image.shape == (20, 10, 3)  # (height, width, channels)
    assert gen.image.dtype == np.uint8

    # Test square image
    gen2 = ImageGenerator(50, 50)
    assert gen2.width == 50
    assert gen2.height == 50
    assert gen2.image.shape == (50, 50, 3)


def test_constructor_invalid_dimensions() -> None:
    """
    Test that constructor rejects invalid dimensions.

    Design decision: Test boundary conditions (zero, negative) separately
    to provide clear error messages.
    """
    # Zero width
    with pytest.raises(ValueError, match="Width and height must be positive"):
        ImageGenerator(0, 10)

    # Zero height
    with pytest.raises(ValueError, match="Width and height must be positive"):
        ImageGenerator(10, 0)

    # Negative width
    with pytest.raises(ValueError, match="Width and height must be positive"):
        ImageGenerator(-5, 10)

    # Negative height
    with pytest.raises(ValueError, match="Width and height must be positive"):
        ImageGenerator(10, -5)


def test_properties_are_immutable() -> None:
    """
    Test that width and height properties are read-only.

    Design decision: Properties should be immutable to prevent dimension
    mismatches that would corrupt the image array.
    """
    gen = ImageGenerator(100, 200)

    # Attempting to set width should fail
    with pytest.raises(AttributeError):
        gen.width = 50  # type: ignore

    # Attempting to set height should fail
    with pytest.raises(AttributeError):
        gen.height = 100  # type: ignore

    # Original values unchanged
    assert gen.width == 100
    assert gen.height == 200


# ============================================================================
# 2. Direct Pixel Assignment Tests
# ============================================================================


def test_set_pixel_valid_coordinates() -> None:
    """
    Test that set_pixel correctly assigns RGB values at valid coordinates.

    Design decision: Test corners and center to cover different array positions.
    """
    gen = ImageGenerator(10, 10)

    # Set top-left corner (0, 0)
    gen.set_pixel(0, 0, 255, 0, 0)  # Red
    assert np.array_equal(gen.image[0, 0], [255, 0, 0])

    # Set bottom-right corner (9, 9)
    gen.set_pixel(9, 9, 0, 255, 0)  # Green
    assert np.array_equal(gen.image[9, 9], [0, 255, 0])

    # Set center (5, 5)
    gen.set_pixel(5, 5, 0, 0, 255)  # Blue
    assert np.array_equal(gen.image[5, 5], [0, 0, 255])

    # Verify get_pixel returns same values
    assert gen.get_pixel(0, 0) == (255, 0, 0)
    assert gen.get_pixel(9, 9) == (0, 255, 0)
    assert gen.get_pixel(5, 5) == (0, 0, 255)


def test_set_pixel_out_of_bounds() -> None:
    """
    Test that set_pixel rejects out-of-bounds coordinates.

    Design decision: Test all boundary violations (negative, too large)
    to ensure complete bounds checking.
    """
    gen = ImageGenerator(10, 10)

    # X coordinate too large
    with pytest.raises(IndexError, match="out of bounds"):
        gen.set_pixel(10, 5, 255, 0, 0)

    # Y coordinate too large
    with pytest.raises(IndexError, match="out of bounds"):
        gen.set_pixel(5, 10, 255, 0, 0)

    # Negative X
    with pytest.raises(IndexError, match="out of bounds"):
        gen.set_pixel(-1, 5, 255, 0, 0)

    # Negative Y
    with pytest.raises(IndexError, match="out of bounds"):
        gen.set_pixel(5, -1, 255, 0, 0)


def test_set_pixel_invalid_rgb_values() -> None:
    """
    Test that set_pixel rejects RGB values outside [0, 255] range.

    Design decision: Test both boundaries (negative and > 255) for each channel.
    """
    gen = ImageGenerator(10, 10)

    # Red value too large
    with pytest.raises(ValueError, match="RGB values must be in range 0-255"):
        gen.set_pixel(5, 5, 256, 128, 128)

    # Green value negative
    with pytest.raises(ValueError, match="RGB values must be in range 0-255"):
        gen.set_pixel(5, 5, 128, -1, 128)

    # Blue value too large
    with pytest.raises(ValueError, match="RGB values must be in range 0-255"):
        gen.set_pixel(5, 5, 128, 128, 300)


# ============================================================================
# 3. Direct Byte Assignment Tests
# ============================================================================


def test_set_byte_valid_index() -> None:
    """
    Test that set_byte correctly assigns values at valid indices.

    Design decision: Test first byte, last byte, and middle byte to cover
    different positions in the flattened array.
    """
    gen = ImageGenerator(10, 10)  # 10*10*3 = 300 bytes

    # Set first byte (R channel of first pixel)
    gen.set_byte(0, 255)
    assert gen.get_byte(0) == 255
    assert gen.image[0, 0, 0] == 255

    # Set last byte (B channel of last pixel)
    gen.set_byte(299, 128)
    assert gen.get_byte(299) == 128
    assert gen.image[9, 9, 2] == 128

    # Set middle byte
    gen.set_byte(150, 64)
    assert gen.get_byte(150) == 64


def test_set_byte_out_of_bounds() -> None:
    """
    Test that set_byte rejects indices outside valid range.

    Design decision: Test boundary conditions (negative, at limit, beyond limit).
    """
    gen = ImageGenerator(10, 10)  # 300 bytes (0-299)

    # Index at byte_count (one past end)
    with pytest.raises(IndexError, match="out of bounds"):
        gen.set_byte(300, 128)

    # Index beyond byte_count
    with pytest.raises(IndexError, match="out of bounds"):
        gen.set_byte(1000, 128)

    # Negative index
    with pytest.raises(IndexError, match="out of bounds"):
        gen.set_byte(-1, 128)


def test_set_byte_invalid_value() -> None:
    """
    Test that set_byte rejects byte values outside [0, 255] range.

    Design decision: Byte values must be valid uint8 range.
    """
    gen = ImageGenerator(10, 10)

    # Value too large
    with pytest.raises(ValueError, match="Byte value must be in range 0-255"):
        gen.set_byte(0, 256)

    # Negative value
    with pytest.raises(ValueError, match="Byte value must be in range 0-255"):
        gen.set_byte(0, -1)

    # Way out of range
    with pytest.raises(ValueError, match="Byte value must be in range 0-255"):
        gen.set_byte(0, 1000)


# ============================================================================
# 4. Recursive Operations Tests
# ============================================================================


def test_set_pixel_recursive_receives_correct_parameters() -> None:
    """
    Test that recursive pixel function receives correct x, y, and image array.

    Design decision: Verify function is called with correct parameters,
    allowing it to reference other pixels.
    """
    gen = ImageGenerator(10, 10)

    # Set a reference pixel
    gen.set_pixel(0, 0, 100, 150, 200)

    # Track what parameters the function receives
    received_params: list[Tuple[int, int, Tuple[int, int, int]]] = []

    def capture_params(x: int, y: int, img: np.ndarray) -> Tuple[int, int, int]:
        received_params.append((x, y, img.shape))
        # Return valid RGB based on reference pixel
        if x > 0:
            return tuple(img[y, x - 1])  # type: ignore
        return (255, 255, 255)

    # Call recursive function
    gen.set_pixel_recursive(5, 5, capture_params)

    # Verify correct parameters were passed
    assert len(received_params) == 1
    assert received_params[0][0] == 5  # x coordinate
    assert received_params[0][1] == 5  # y coordinate
    assert received_params[0][2] == (10, 10, 3)  # image shape


def test_set_byte_recursive_receives_correct_parameters() -> None:
    """
    Test that recursive byte function receives correct index and flat array.

    Design decision: Verify function can access previous bytes for
    sequential pattern generation.
    """
    gen = ImageGenerator(10, 10)

    # Set some reference bytes
    gen.set_byte(0, 100)
    gen.set_byte(1, 150)

    # Track what parameters the function receives
    received_params: list[Tuple[int, int]] = []

    def capture_params(idx: int, flat: np.ndarray) -> int:
        received_params.append((idx, len(flat)))
        # Return valid value based on previous byte
        if idx > 0:
            return int(flat[idx - 1])
        return 255

    # Call recursive function
    gen.set_byte_recursive(10, capture_params)

    # Verify correct parameters were passed
    assert len(received_params) == 1
    assert received_params[0][0] == 10  # byte index
    assert received_params[0][1] == 300  # flat array length (10*10*3)


# ============================================================================
# 5. File I/O Tests
# ============================================================================


def test_save_creates_valid_png(temp_image_path: Path) -> None:
    """
    Test that save() creates a valid PNG file that can be loaded.

    Design decision: Verify file exists, can be loaded by PIL, and has
    correct dimensions and pixel values.

    Args:
        temp_image_path: pytest fixture providing temporary file path
    """
    # Create and fill image
    gen = ImageGenerator(50, 30)
    gen.set_pixel(10, 10, 255, 128, 64)
    gen.set_pixel(20, 20, 64, 128, 255)

    # Save to file
    gen.save(str(temp_image_path))

    # Verify file exists
    assert os.path.exists(temp_image_path)

    # Verify file can be loaded by PIL
    img = Image.open(temp_image_path)
    assert img.size == (50, 30)  # PIL uses (width, height)
    assert img.mode == "RGB"

    # Verify pixel values preserved
    img_array = np.array(img)
    assert np.array_equal(img_array[10, 10], [255, 128, 64])
    assert np.array_equal(img_array[20, 20], [64, 128, 255])


# ============================================================================
# 6. Edge Cases - Week 2
# ============================================================================


def test_minimum_size_image() -> None:
    """
    Test 1x1 pixel image (minimum valid size).

    Design decision: Smallest possible image should work correctly.
    Edge case important for correlation/Monte Carlo functions.
    """
    gen = ImageGenerator(1, 1)
    assert gen.width == 1
    assert gen.height == 1
    assert gen.byte_count == 3  # 1 pixel × 3 channels

    # Should be able to set the single pixel
    gen.set_pixel(0, 0, 255, 128, 64)
    assert gen.get_pixel(0, 0) == (255, 128, 64)


def test_very_large_image_dimensions() -> None:
    """
    Test that large (but reasonable) images can be created.

    Design decision: Verify no overflow issues with large dimensions.
    Don't make it TOO large to avoid slow tests/memory issues.
    """
    # 4K resolution image (common size)
    gen = ImageGenerator(3840, 2160)
    assert gen.width == 3840
    assert gen.height == 2160
    assert gen.byte_count == 3840 * 2160 * 3  # ~25 million bytes

    # Verify corners accessible
    gen.set_pixel(0, 0, 255, 0, 0)
    gen.set_pixel(3839, 2159, 0, 255, 0)
    assert gen.get_pixel(0, 0) == (255, 0, 0)
    assert gen.get_pixel(3839, 2159) == (0, 255, 0)


def test_boundary_rgb_values() -> None:
    """
    Test boundary RGB values (0 and 255).

    Design decision: Ensure edge values work correctly without overflow.
    """
    gen = ImageGenerator(10, 10)

    # Minimum values (black)
    gen.set_pixel(0, 0, 0, 0, 0)
    assert gen.get_pixel(0, 0) == (0, 0, 0)

    # Maximum values (white)
    gen.set_pixel(5, 5, 255, 255, 255)
    assert gen.get_pixel(5, 5) == (255, 255, 255)

    # Mixed boundaries
    gen.set_pixel(9, 9, 0, 255, 0)
    assert gen.get_pixel(9, 9) == (0, 255, 0)


def test_pixel_byte_correspondence() -> None:
    """
    Test that pixel and byte operations access the same underlying data.

    Design decision: Setting pixel (x, y) should affect bytes at correct indices.
    Verifies row-major ordering and RGB interleaving.
    """
    gen = ImageGenerator(10, 10)

    # Set pixel at (2, 3) - this is row 3, column 2
    gen.set_pixel(2, 3, 100, 150, 200)

    # Calculate byte indices: (row * width + col) * 3 + channel
    # Row 3, col 2 = (3 * 10 + 2) * 3 = 32 * 3 = 96
    base_index = (3 * 10 + 2) * 3

    assert gen.get_byte(base_index + 0) == 100  # R
    assert gen.get_byte(base_index + 1) == 150  # G
    assert gen.get_byte(base_index + 2) == 200  # B


def test_fill_method_with_boundary_values() -> None:
    """
    Test fill() with extreme RGB values.

    Design decision: Ensure fill works correctly with edge values.
    """
    gen = ImageGenerator(20, 20)

    # Fill with black
    gen.fill(0, 0, 0)
    assert np.all(gen.image == 0)

    # Fill with white
    gen.fill(255, 255, 255)
    assert np.all(gen.image == 255)


def test_recursive_function_edge_cases() -> None:
    """
    Test recursive functions with edge conditions.

    Design decision: Recursive function should handle first pixel/byte
    (no predecessors) without error.
    """
    gen = ImageGenerator(5, 5)

    # Recursive pixel function that references left neighbor
    def left_or_white(x: int, y: int, img: np.ndarray) -> Tuple[int, int, int]:
        if x == 0:
            return (255, 255, 255)  # No left neighbor
        return tuple(img[y, x - 1])  # type: ignore

    # First column should all be white
    for y in range(5):
        gen.set_pixel_recursive(0, y, left_or_white)
        assert gen.get_pixel(0, y) == (255, 255, 255)


def test_get_methods_match_set_methods() -> None:
    """
    Test that get_pixel and get_byte return exactly what was set.

    Design decision: Round-trip test - set then get should be identity.
    """
    gen = ImageGenerator(10, 10)

    # Test multiple positions
    test_cases = [(0, 0, 100, 50, 25), (5, 5, 255, 128, 64), (9, 9, 0, 0, 0)]

    for x, y, r, g, b in test_cases:
        gen.set_pixel(x, y, r, g, b)
        assert gen.get_pixel(x, y) == (r, g, b)


def test_byte_operations_on_large_indices() -> None:
    """
    Test byte operations at the end of large arrays.

    Design decision: Verify no overflow or index calculation errors
    with large byte counts.
    """
    gen = ImageGenerator(1000, 1000)  # 3 million bytes
    last_byte = gen.byte_count - 1

    # Set and get last byte
    gen.set_byte(last_byte, 123)
    assert gen.get_byte(last_byte) == 123

    # Set and get middle byte
    middle = gen.byte_count // 2
    gen.set_byte(middle, 45)
    assert gen.get_byte(middle) == 45


def test_clear_resets_all_pixels() -> None:
    """
    Test that clear() truly resets entire image.

    Design decision: Clear should work regardless of previous state.
    """
    gen = ImageGenerator(50, 50)

    # Fill with random values
    for i in range(100):  # Set some random pixels
        x, y = i % 50, i // 50
        gen.set_pixel(x, y, 255, 255, 255)

    # Clear
    gen.clear()

    # Verify all black
    assert np.all(gen.image == 0)

    # Verify via pixel access too
    assert gen.get_pixel(25, 25) == (0, 0, 0)


def test_recursive_function_invalid_return_values() -> None:
    """
    Test that recursive functions with invalid returns are caught.

    Design decision: Validate recursive function outputs to prevent corruption.
    """
    gen = ImageGenerator(10, 10)

    # Function returning invalid RGB
    def invalid_rgb(x: int, y: int, img: np.ndarray) -> Tuple[int, int, int]:
        return (256, 0, 0)  # Invalid: > 255

    with pytest.raises(ValueError, match="invalid RGB values"):
        gen.set_pixel_recursive(5, 5, invalid_rgb)

    # Function returning invalid byte
    def invalid_byte(idx: int, flat: np.ndarray) -> int:
        return 300  # Invalid: > 255

    with pytest.raises(ValueError, match="invalid byte value"):
        gen.set_byte_recursive(10, invalid_byte)


# ============================================================================
# Additional Utility Method Tests
# ============================================================================


def test_fill_method() -> None:
    """
    Test that fill() sets all pixels to specified color.

    Design decision: Fill is a convenience method, so verify it actually
    sets all pixels uniformly.
    """
    gen = ImageGenerator(20, 20)
    gen.fill(100, 150, 200)

    # Check all pixels have the fill color
    assert np.all(gen.image[:, :, 0] == 100)  # Red channel
    assert np.all(gen.image[:, :, 1] == 150)  # Green channel
    assert np.all(gen.image[:, :, 2] == 200)  # Blue channel


def test_clear_method() -> None:
    """
    Test that clear() resets image to all black pixels.

    Design decision: Clear should be equivalent to fill(0, 0, 0).
    """
    gen = ImageGenerator(20, 20)

    # Set some non-zero values
    gen.fill(100, 100, 100)

    # Clear should reset to zero
    gen.clear()

    # Verify all pixels are black
    assert np.all(gen.image == 0)


def test_byte_count_property() -> None:
    """
    Test that byte_count property returns correct total.

    Design decision: byte_count = width × height × 3 (RGB channels).
    """
    gen = ImageGenerator(10, 20)
    assert gen.byte_count == 10 * 20 * 3  # 600 bytes

    gen2 = ImageGenerator(100, 100)
    assert gen2.byte_count == 100 * 100 * 3  # 30,000 bytes
