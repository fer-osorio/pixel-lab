from math import cos, sqrt

from pixel_lab import ImageGenerator


def func_on_image(x: int):
    y = x / 16
    return int(y * sqrt(y) * cos(y / 32))


def direct_byte_assignment():
    """Example: Create a pattern using direct byte assignment."""
    print("\n=== Example: Direct Byte Assignment ===")

    gen = ImageGenerator(1152, 512)

    # Assigning each byte the value of its index module 256
    j = 0
    k = -1
    offset = 0
    for i in range(0, gen.byte_count, 3):
        j = (i // 3) & 255
        if j == 0:
            k += 1
            offset = func_on_image(k)
        gen.set_byte(i, (j + offset) & 255)
        gen.set_byte(i + 1, (j + offset) & 255)
        gen.set_byte(i + 2, (j + offset) & 255)

    gen.save("direct_byte_assignment.png")


if __name__ == "__main__":
    direct_byte_assignment()
