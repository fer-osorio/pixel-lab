from image_toolkit import ImageGenerator
import numpy as np

def recursive_byte_assignment():
    """Example: Create a pattern where each byte depends on previous bytes."""
    print("\n=== Example 4: Recursive Byte Assignment (XOR Pattern) ===")

    gen = ImageGenerator(256, 256)

    # Define recursive function: Fibonacci sequence
    def xor_pattern(idx: int, flat: np.ndarray) -> int:
        # Initializing sequence
        if idx == 0:
            return 0
        if idx == 1:
            return 1
        # Fibonacci rule
        return (int(flat[idx - 1]) + int(flat[idx - 2])) & 255

    # Fill the image recursively at the byte level
    for i in range(gen.byte_count):
        gen.set_byte_recursive(i, xor_pattern)

    gen.save("recursive_byte_assignment.png")

if __name__ == "__main__":
    recursive_byte_assignment()
