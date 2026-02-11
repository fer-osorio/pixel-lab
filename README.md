# HPPixelLab

## Project Overview
HPPixelLab is a library designed for generating and manipulating images efficiently and with high precision. The name "HPPixelLab" stands for "High Precision Pixel Laboratory".

## Features
- Image generation from scratch.
- Metrics calculation for image analysis.
- Compatibility with various image formats.
- Lightweight and efficient performance.

## Installation
To install HPPixelLab, simply clone the repository and install the dependencies:

```bash
git clone https://github.com/fer-osorio/HPPixelLab.git
cd HPPixelLab
pip install -r requirements.txt
```

## Usage Examples

### Generating an Image
```python
from image_generator import ImageGenerator

gen = ImageGenerator(256, 256)
# Assigning each byte the value of its index module 256
for i in range(gen.byte_count):
  gen.set_byte(i, i & 255)

gen.save("direct_byte_assignment.png")
```

### Calculating Image Metrics
```python
from image_metrics import ImageMetrics

metrics = ImageMetrics("image.png")
entropy_blue = metrics.entropy("blue")
print(f'Image entropy in blue channel: {entropy_blue}')
```

# Requirements

- Python 3.x
- numpy >= 1.20.0
- pillow >= 9.0.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
