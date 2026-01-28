"""
Pixel-Level Image Generator
A flexible image generation system with fine-grained control over pixel and byte values.

Design decisions:
1. NumPy for computational efficiency and flexible array views
2. PIL for robust image I/O operations
3. Immutable width/height to prevent dimension mismatches
4. Bounds checking to prevent index errors
5. Support for both functional and method-based workflows
"""

import numpy as np
from PIL import Image
from typing import Callable, Tuple, Optional


class ImageGenerator:
    """
    A class for generating images with precise pixel and byte-level control.

    The image can be manipulated through four different paradigms:
    1. Direct pixel assignment by coordinates (x, y)
    2. Direct byte assignment by index
    3. Recursive pixel assignment using previously set pixels
    4. Recursive byte assignment using previously set bytes
    """

    def __init__(self, width: int, height: int):
        """
        Initialize the image generator.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Design decision: Store dimensions separately for easy access and validation.
        Using uint8 dtype ensures valid RGB values (0-255) and memory efficiency.
        """
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive integers")

        self._width = width
        self._height = height

        # Primary data structure: 3D NumPy array (height, width, channels)
        # Shape is (height, width, 3) to match image coordinate convention
        self.image = np.zeros((height, width, 3), dtype=np.uint8)

    @property
    def width(self) -> int:
        """Image width (immutable after creation)"""
        return self._width

    @property
    def height(self) -> int:
        """Image height (immutable after creation)"""
        return self._height

    @property
    def byte_count(self) -> int:
        """Total number of bytes in the image (width * height * 3)"""
        return self._width * self._height * 3

    # ========================================================================
    # METHOD 1: Direct Pixel Assignment by Coordinates
    # ========================================================================

    def set_pixel(self, x: int, y: int, r: int, g: int, b: int) -> None:
        """
        Set a pixel value by coordinates (matrix view).

        Args:
            x: X-coordinate (column), 0-indexed from left
            y: Y-coordinate (row), 0-indexed from top
            r, g, b: RGB color values (0-255)

        Design decision: Use (x, y) convention matching screen coordinates,
        but internally access as [y, x] for NumPy row-major order.
        """
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise IndexError(f"Pixel coordinates ({x}, {y}) out of bounds")

        if not all(0 <= val <= 255 for val in [r, g, b]):
            raise ValueError("RGB values must be in range 0-255")

        # Note: NumPy uses [row, column] = [y, x] indexing
        self.image[y, x] = [r, g, b]

    def get_pixel(self, x: int, y: int) -> Tuple[int, int, int]:
        """
        Get a pixel's RGB values by coordinates.

        Returns:
            Tuple of (r, g, b) values
        """
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise IndexError(f"Pixel coordinates ({x}, {y}) out of bounds")

        pixel = self.image[y, x]
        return int(pixel[0]), int(pixel[1]), int(pixel[2])

    # ========================================================================
    # METHOD 2: Direct Byte Assignment by Index
    # ========================================================================

    def set_byte(self, byte_index: int, value: int) -> None:
        """
        Set a single byte value by index (flat array view).

        Args:
            byte_index: Index in the flattened byte array (0 to width*height*3-1)
            value: Byte value (0-255)

        Design decision: Create a flat view on-demand rather than storing it
        permanently. This ensures the view is always in sync with the image
        and doesn't waste memory.

        Byte ordering: The flat array follows row-major order with RGB interleaved:
        [R0, G0, B0, R1, G1, B1, ...] where 0, 1, etc. are pixel indices.
        """
        if not (0 <= byte_index < self.byte_count):
            raise IndexError(f"Byte index {byte_index} out of bounds (max: {self.byte_count - 1})")

        if not (0 <= value <= 255):
            raise ValueError("Byte value must be in range 0-255")

        # Create a flat view of the image array
        # Design note: reshape(-1) creates a 1D view without copying data
        flat_view = self.image.reshape(-1)
        flat_view[byte_index] = value

    def get_byte(self, byte_index: int) -> int:
        """
        Get a single byte value by index.

        Returns:
            Byte value (0-255)
        """
        if not (0 <= byte_index < self.byte_count):
            raise IndexError(f"Byte index {byte_index} out of bounds (max: {self.byte_count - 1})")

        flat_view = self.image.reshape(-1)
        return int(flat_view[byte_index])

    # ========================================================================
    # METHOD 3: Recursive Pixel Assignment by Coordinates
    # ========================================================================

    def set_pixel_recursive(self, x: int, y: int,
                           func: Callable[[int, int, np.ndarray], Tuple[int, int, int]]) -> None:
        """
        Set a pixel using a function that can reference other pixels (matrix view).

        Args:
            x: X-coordinate of target pixel
            y: Y-coordinate of target pixel
            func: Function that takes (x, y, image_array) and returns (r, g, b)
                  The image_array parameter allows the function to read other pixels.

        Design decision: Pass the entire image array to the function for maximum
        flexibility. The function can implement any recursive logic, such as:
        - Averaging neighboring pixels
        - Creating fractals based on position
        - Building cellular automata patterns

        Example function:
            def copy_left_pixel(x, y, img):
                if x == 0:
                    return (0, 0, 0)  # Left edge is black
                return tuple(img[y, x-1])  # Copy pixel to the left
        """
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise IndexError(f"Pixel coordinates ({x}, {y}) out of bounds")

        # Call the user-provided function to compute RGB values
        r, g, b = func(x, y, self.image)

        # Validate and set the computed values
        if not all(0 <= val <= 255 for val in [r, g, b]):
            raise ValueError(f"Function returned invalid RGB values: ({r}, {g}, {b})")

        self.image[y, x] = [r, g, b]

    # ========================================================================
    # METHOD 4: Recursive Byte Assignment by Index
    # ========================================================================

    def set_byte_recursive(self, byte_index: int,
                          func: Callable[[int, np.ndarray], int]) -> None:
        """
        Set a byte using a function that can reference other bytes (flat array view).

        Args:
            byte_index: Index of target byte
            func: Function that takes (byte_index, flat_array) and returns byte value
                  The flat_array parameter allows the function to read other bytes.

        Design decision: Provide flat array view for byte-level operations.
        This is useful for creating patterns based on byte sequences, like:
        - XOR-based patterns
        - Byte-level cellular automata
        - Checksum-like recursive patterns

        Example function:
            def xor_previous_bytes(idx, flat):
                if idx == 0:
                    return 128
                return flat[idx-1] ^ 0xFF  # XOR previous byte with 255
        """
        if not (0 <= byte_index < self.byte_count):
            raise IndexError(f"Byte index {byte_index} out of bounds (max: {self.byte_count - 1})")

        # Create flat view for the function to read from
        flat_view = self.image.reshape(-1)

        # Call the user-provided function to compute the byte value
        value = func(byte_index, flat_view)

        # Validate and set the computed value
        if not (0 <= value <= 255):
            raise ValueError(f"Function returned invalid byte value: {value}")

        flat_view[byte_index] = value

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def fill(self, r: int, g: int, b: int) -> None:
        """
        Fill the entire image with a solid color.

        Design decision: Use NumPy's vectorized assignment for efficiency
        rather than looping through pixels.
        """
        if not all(0 <= val <= 255 for val in [r, g, b]):
            raise ValueError("RGB values must be in range 0-255")

        self.image[:, :] = [r, g, b]

    def clear(self) -> None:
        """Reset the image to all black pixels."""
        self.image.fill(0)

    def save(self, filename: str) -> None:
        """
        Save the image to a file.

        Args:
            filename: Output path (format determined by extension: .png, .jpg, etc.)

        Design decision: Use PIL for I/O because it handles format conversion,
        color space management, and compression automatically.
        """
        # Convert NumPy array to PIL Image
        # 'RGB' mode ensures correct color interpretation
        img = Image.fromarray(self.image, mode='RGB')
        img.save(filename)
        print(f"Image saved to: {filename}")

    def show(self) -> None:
        """
        Display the image using the default image viewer.

        Design decision: Useful for quick visualization during development.
        """
        img = Image.fromarray(self.image, mode='RGB')
        img.show()

    def get_numpy_array(self) -> np.ndarray:
        """
        Get a copy of the underlying NumPy array.

        Returns:
            Copy of the image array (shape: height x width x 3)

        Design decision: Return a copy to prevent external modification
        of internal state. Use this if you need to perform bulk NumPy operations.
        """
        return self.image.copy()
