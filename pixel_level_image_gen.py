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


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_1_direct_pixel_assignment():
    """Example 1: Create a simple gradient using direct pixel assignment."""
    print("\n=== Example 1: Direct Pixel Assignment (Gradient) ===")

    gen = ImageGenerator(256, 256)

    # Create a horizontal red gradient
    for x in range(gen.width):
        for y in range(gen.height):
            # Red and green increases with x and y, blue stay at 255
            gen.set_pixel(x, y, x, y, 255)

    gen.save("example1_gradient.png")


def example_2_direct_byte_assignment():
    """Example 2: Create a pattern using direct byte assignment."""
    print("\n=== Example 2: Direct Byte Assignment (Incresing function) ===")

    gen = ImageGenerator(256, 256)

    # Assigning each byte the value of its index
    for i in range(gen.byte_count):
        gen.set_byte(i, i & 255)

    gen.save("example2_incresing.png")


def example_3_recursive_pixel_assignment():
    """Example 3: Create a pattern where each pixel depends on its neighbors."""
    print("\n=== Example 3: Recursive Pixel Assignment (Neighbor Average) ===")

    gen = ImageGenerator(256, 256)
    div = 63

    # Set a bright pixel in the center as a seed
    for y in range(gen.height):
        for x in range(gen.width):
            if (x & div) == 0 and (y & div) == 0:
                gen.set_pixel(x, y, 255, 255, 255)

    # Define a function that averages the left and top neighbors
    def average_neighbors(x: int, y: int, img: np.ndarray) -> Tuple[int, int, int]:
        if x == 0 and y == 0:
            return (255, 255, 255)  # Top-left corner is white

        # Get top neighbor or default to black
        top_r = int(img[y-1, x, 0]) if y > 0 else 0
        top_g = int(img[y-1, x, 1]) if y > 0 else 0
        top_b = int(img[y-1, x, 2]) if y > 0 else 0

        # Get left neighbor or default to black
        left_r = int(img[y, x-1, 0]) if x > 0 else 0
        left_g = int(img[y, x-1, 1]) if x > 0 else 0
        left_b = int(img[y, x-1, 2]) if x > 0 else 0

        # Get top neighbor or default to black
        topleft_r = int(img[y-1, x-1, 0]) if y > 0 and x > 0 else 0
        topleft_g = int(img[y-1, x-1, 1]) if y > 0 and x > 0 else 0
        topleft_b = int(img[y-1, x-1, 2]) if y > 0 and x > 0 else 0

        # Average the neighbors
        r = (left_r + top_r + topleft_r) // 3
        g = (left_g + top_g + topleft_g) // 3
        b = (left_b + top_b + topleft_b) // 3

        return (r, g, b)

    # Fill the image recursively, row by row, left to right
    for y in range(gen.height):
        for x in range(gen.width):
            if (x & div) != 0 or (y & div) != 0:
                gen.set_pixel_recursive(x, y, average_neighbors)

    gen.save("example3_recursive_pixels.png")


def example_4_recursive_byte_assignment():
    """Example 4: Create a pattern where each byte depends on previous bytes."""
    print("\n=== Example 4: Recursive Byte Assignment (XOR Pattern) ===")

    gen = ImageGenerator(256, 256)

    # Define a function that creates an XOR-based pattern
    def xor_pattern(idx: int, flat: np.ndarray) -> int:
        if idx == 0:
            return 255  # First byte is white

        # Create pattern based on XOR of previous byte and position
        prev_value = int(flat[idx - 1])
        position_factor = idx % 256

        return (prev_value ^ position_factor) % 256

    # Fill the image recursively at the byte level
    for i in range(gen.byte_count):
        gen.set_byte_recursive(i, xor_pattern)

    gen.save("example4_recursive_bytes.png")


def example_5_complex_fractal():
    """Example 5: Create a simple Sierpinski-like triangle pattern."""
    print("\n=== Example 5: Complex Pattern (Sierpinski-style) ===")

    gen = ImageGenerator(512, 512)

    # Define a rule based on bitwise operations on coordinates
    def sierpinski_rule(x: int, y: int, img: np.ndarray) -> Tuple[int, int, int]:
        # Sierpinski pattern using bitwise AND
        if (x & y) == 0:
            return (255, 255, 255)  # White
        else:
            return (0, 0, 128)  # Dark blue

    # Generate the pattern
    for y in range(gen.height):
        for x in range(gen.width):
            gen.set_pixel_recursive(x, y, sierpinski_rule)

    gen.save("example5_sierpinski.png")


if __name__ == "__main__":
    print("Pixel-Level Image Generator - Examples")
    print("=" * 50)

    # Run all examples
    example_1_direct_pixel_assignment()
    example_2_direct_byte_assignment()
    example_3_recursive_pixel_assignment()
    example_4_recursive_byte_assignment()
    example_5_complex_fractal()

    print("\n" + "=" * 50)
    print("All examples completed! Check the generated PNG files.")
