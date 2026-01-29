# ============================================================================
# USAGE EXAMPLES
# ============================================================================

import sys
import os
# Get the absolute path of the directory you want to import from
# This example goes up one directory level, then into 'src/image_toolkit'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
module_dir = os.path.join(parent_dir, 'src/image_toolkit')

# Insert the directory to the beginning of the search path
sys.path.insert(0, module_dir)

import numpy as np
from typing import Callable, Tuple, Optional
import generator as Gen

def example_1_direct_pixel_assignment():
    """Example 1: Create a simple gradient using direct pixel assignment."""
    print("\n=== Example 1: Direct Pixel Assignment (Gradient) ===")

    gen = Gen.ImageGenerator(256, 256)

    # Create a horizontal and vertical gradient
    for x in range(gen.width):
        for y in range(x, gen.height):
            # Red and green increases with x and y, blue stay at 255
            gen.set_pixel(x, y, x, y, x)

    for y in range(gen.height):
        for x in range(y, gen.width):
            # Red and green increases with x and y, blue stay at 255
            gen.set_pixel(x, y, x, y, 255 - y)

    gen.save("example1_gradient.png")


def example_2_direct_byte_assignment():
    """Example 2: Create a pattern using direct byte assignment."""
    print("\n=== Example 2: Direct Byte Assignment (Incresing function) ===")

    gen = Gen.ImageGenerator(256, 256)

    # Assigning each byte the value of its index
    for i in range(gen.byte_count):
        gen.set_byte(i, i & 255)

    gen.save("example2_incresing.png")


def example_3_recursive_pixel_assignment():
    """Example 3: Create a pattern where each pixel depends on its neighbors."""
    print("\n=== Example 3: Recursive Pixel Assignment (Neighbor Average) ===")

    gen = Gen.ImageGenerator(256, 256)
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

    gen = Gen.ImageGenerator(256, 256)

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
    sz = 512
    szdiv = (sz + 255) // 256;
    gen = Gen.ImageGenerator(sz, sz)

    # Define a rule based on bitwise operations on coordinates
    def sierpinski_rule(x: int, y: int, img: np.ndarray) -> Tuple[int, int, int]:
        # Sierpinski pattern using bitwise AND
        if (x & y) == 0:
            return (255, 255, 255)  # White
        else:
            return (x // szdiv, y // szdiv, 255) # Cool background

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
