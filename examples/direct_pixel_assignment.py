from image_toolkit import ImageGenerator

def direct_pixel_assignment():
    """Example: Create a simple gradient using direct pixel assignment."""
    print("\n=== Example 1: Direct Pixel Assignment (Gradient) ===")

    gen = ImageGenerator(256, 256)

    # Create gradient from diagonal to bottom
    for x in range(gen.width):
        for y in range(x, gen.height):
            # Red and blue increases with x, blue increases with y
            gen.set_pixel(x, y, x, y, x)

    # Create gradient from diagonal to rigth
    for y in range(gen.height):
        for x in range(y, gen.width):
            # Red increases with x, green increases with y, blue decreases with y
            gen.set_pixel(x, y, x, y, 255 - y)

    gen.save("direct_pixel_assignment.png")

if __name__ == "__main__":
    direct_pixel_assignment()
