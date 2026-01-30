from image_toolkit import ImageGenerator

def direct_byte_assignment():
    """Example: Create a pattern using direct byte assignment."""
    print("\n=== Example 2: Direct Byte Assignment (Incresing function) ===")

    gen = ImageGenerator(256, 256)

    # Assigning each byte the value of its index module 256
    for i in range(gen.byte_count):
        gen.set_byte(i, i & 255)

    gen.save("direct_byte_assignment.png")

if __name__ == "__main__":
    direct_byte_assignment()
