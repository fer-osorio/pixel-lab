# HPPixelLab

## Project Overview
HPPixelLab is a powerful library designed for generating and manipulating images efficiently. It provides an intuitive interface for common image processing tasks, harnessing the capabilities of modern computing.

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

generator = ImageGenerator()
image = generator.create_image(width=256, height=256, color='blue')
image.show()
```

### Calculating Image Metrics
```python
from image_metrics import ImageMetrics

metrics = ImageMetrics(image)
size = metrics.get_size()
print(f'Image Size: {size}')
```

## API Reference

### ImageGenerator Class
- **Methods**
  - `create_image(width: int, height: int, color: str)`: Generates an image with the specified width, height, and background color.

### ImageMetrics Class
- **Methods**
  - `get_size()`: Returns the dimensions of the image.
  - `get_average_color()`: Calculates the average color of the image.

For more detailed information, please refer to the individual class documentation.