from typing import Tuple

import numpy as np

from pixel_lab import ImageGenerator


def recursive_pixel_assignment():
    """Example: Create a pattern where each pixel depends on its neighbors."""
    print("\n=== Example: Recursive Pixel Assignment (Neighbor Average) ===")

    gen = ImageGenerator(256, 256)
    div = 15  # Sets the position of the bright pixels, div has the form 2^k - 1

    # Set a bright pixel in a greed separated by div bits
    for y in range(gen.height):
        for x in range(gen.width):
            if (x & div) == 0 and (y & div) == 0:
                gen.set_pixel(x, y, 255, 255, 255)

    # Define a function that averages the left and top neighbors
    def average_neighbors(x: int, y: int, img: np.ndarray) -> Tuple[int, int, int]:
        if x == 0 and y == 0:
            return (255, 255, 255)  # Top-left corner is white

        # Get top neighbor or default to black
        top_r = int(img[y - 1, x, 0]) if y > 0 else 0
        top_g = int(img[y - 1, x, 1]) if y > 0 else 0
        top_b = int(img[y - 1, x, 2]) if y > 0 else 0

        # Get left neighbor or default to black
        left_r = int(img[y, x - 1, 0]) if x > 0 else 0
        left_g = int(img[y, x - 1, 1]) if x > 0 else 0
        left_b = int(img[y, x - 1, 2]) if x > 0 else 0

        # Get top-left neighbor or default to black
        topleft_r = int(img[y - 1, x - 1, 0]) if y > 0 and x > 0 else 0
        topleft_g = int(img[y - 1, x - 1, 1]) if y > 0 and x > 0 else 0
        topleft_b = int(img[y - 1, x - 1, 2]) if y > 0 and x > 0 else 0

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

    gen.save("recursive_pixel_assignment.png")


if __name__ == "__main__":
    recursive_pixel_assignment()
