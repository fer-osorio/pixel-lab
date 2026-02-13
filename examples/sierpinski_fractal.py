from typing import Tuple

import numpy as np
from pixel_lab import ImageGenerator


def sierpinski_fractal():
    """Example: Create a simple Sierpinski-like triangle pattern."""
    print("\n=== Example: Complex Pattern (Sierpinski-style) ===")
    size = 512
    szdiv = (size + 255) // 256  # Ceil of size / 256
    gen = ImageGenerator(size, size)

    # Define a rule based on bitwise operations on coordinates
    def sierpinski_rule(x: int, y: int, img: np.ndarray) -> Tuple[int, int, int]:
        # Sierpinski pattern using bitwise AND
        if (x & y) == 0:
            return (255, 255, 255)  # White
        else:
            return (x // szdiv, y // szdiv, 255)  # Cool background

    # Generate the pattern
    for y in range(gen.height):
        for x in range(gen.width):
            gen.set_pixel_recursive(x, y, sierpinski_rule)

    gen.save("sierpinski_fractal.png")


if __name__ == "__main__":
    sierpinski_fractal()
