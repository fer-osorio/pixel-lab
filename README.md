# Pixel Lab

A Python toolkit for precise pixel-level image generation and cryptanalytic analysis.

## Overview

Pixel Lab provides fine-grained control over image generation at the pixel and byte level, combined with comprehensive cryptanalytic metrics to validate randomness and statistical properties. Perfect for testing PRNGs, creating algorithmic art, exploring steganography, and understanding image statistics.

## Key Features

### Image Generation (`ImageGenerator`)
- **Direct pixel assignment** by (x, y) coordinates with RGB values
- **Direct byte assignment** by index in flattened array
- **Recursive pixel generation** using functions that reference other pixels
- **Recursive byte generation** using functions that reference other bytes
- Full control over every pixel and byte value

### Cryptanalytic Analysis (`ImageMetrics`)
- **Shannon Entropy** - Measure information density and randomness
- **Chi-Square Test** - Test uniformity of byte distribution
- **Correlation Analysis** - Detect patterns (horizontal, vertical, diagonal)
- **Mean Absolute Deviation** - Simple uniformity metric
- **Monte Carlo π Estimation** - Classic 2D randomness test
- **Comprehensive Visualizations** - Frequency histograms, correlation heatmaps, Monte Carlo plots

## Installation

### Requirements
- Python 3.8+
- NumPy >= 1.20.0
- Pillow >= 9.0.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0

### Install from source
```bash
git clone https://github.com/yourusername/pixel-lab.git
cd pixel-lab
pip install -e . # -e flag installs in developer mode
```

### Install dependencies only
```bash
pip install -r requirements.txt
```

## Quick Start

### Generate an Image

```python
from pixel_lab import ImageGenerator

# Create 256x256 image
gen = ImageGenerator(256, 256)

# Method 1: Set individual pixels
gen.set_pixel(100, 100, 255, 0, 0)  # Red pixel at (100, 100)

# Method 2: Set bytes directly
gen.set_byte(0, 255)  # First byte (red channel of first pixel)

# Method 3: Recursive pixel assignment
def gradient(x, y, img):
    return (x, y, 0)  # Creates gradient pattern

for y in range(gen.height):
    for x in range(gen.width):
        gen.set_pixel_recursive(x, y, gradient)

# Save the result
gen.save("output.png")
```

### Analyze an Image

```python
from pixel_lab import ImageMetrics

# Load and analyze
metrics = ImageMetrics("output.png")

# Get individual metrics
entropy = metrics.entropy(ImageMetrics.RGB.ALL)
chi2 = metrics.chi_square(ImageMetrics.RGB.ALL)
correlation = metrics.correlation(ImageMetrics.RGB.ALL, 'horizontal')

print(f"Entropy: {entropy:.4f} bits")
print(f"Chi-square p-value: {chi2['p_value']:.6f}")
print(f"Correlation: {correlation:+.6f}")

# Generate comprehensive report
print(metrics.summary(ImageMetrics.RGB.ALL, verbose=True))

# Create visualizations
metrics.plot_all(ImageMetrics.RGB.ALL, save_dir='./analysis')
```

### Complete Workflow: Test a PRNG

```python
import numpy as np
from pixel_lab import ImageGenerator, ImageMetrics

# Generate image with PRNG
gen = ImageGenerator(512, 512)
for i in range(gen.byte_count):
    gen.set_byte(i, np.random.randint(0, 256))
gen.save("prng_test.png")

# Analyze randomness
metrics = ImageMetrics("prng_test.png")
analysis = metrics.analyze_all(ImageMetrics.RGB.ALL)

# Check quality
passes = (
    analysis['entropy'] > 7.9 and
    analysis['chi_square']['p_value'] > 0.05 and
    abs(analysis['correlation_horizontal']) < 0.1 and
    analysis['mean_absolute_deviation']['mad_percentage'] < 10.0 and
    analysis['monte_carlo_pi']['error_percentage'] < 5.0
)

print("✓ PRNG passes all tests!" if passes else "✗ PRNG failed tests")
print(metrics.summary(ImageMetrics.RGB.ALL))
```

## Command-Line Interface

### Generate Images
```bash
# Coming soon - see examples/ for now
```

### Analyze Images
```bash
# Basic analysis
python scripts/analyze_image.py image.png

# Analyze specific channel
python scripts/analyze_image.py image.png --channel red

# Generate visualizations
python scripts/analyze_image.py image.png --plot

# Save analysis report and plots
python scripts/analyze_image.py image.png --save-plots ./output --verbose
```

## Examples

Explore the `examples/` directory for detailed tutorials:

- **01_basic_generation.py** - Simple pixel and byte manipulation
- **02_recursive_patterns.py** - Creating fractals and algorithmic art
- **03_prng_testing.py** - Validating pseudo-random number generators
- **04_cryptanalysis.py** - Full cryptanalytic workflow

### Running Examples

Use the automated runner script to execute examples:

```bash
# Run a specific example
cd examples/
./run_examples.sh 01_basic_generation.py

# Run all examples
./run_examples.sh --all

# Run examples by category
./run_examples.sh --pixel       # All pixel-related examples
./run_examples.sh --byte        # All byte-related examples
./run_examples.sh --direct      # All direct assignment examples
./run_examples.sh --recursive   # All recursive examples

# Verbose mode (show full output)
./run_examples.sh --all --verbose

# Get help
./run_examples.sh --help
```

The runner validates scripts, provides colored output, and shows a summary report.

## Use Cases

### PRNG Validation
Test if your pseudo-random number generator produces truly random output by generating an image and analyzing its statistical properties.

### Algorithmic Art
Create generative art using mathematical functions, cellular automata, or fractal algorithms with precise pixel control.

### Steganography Research
Analyze images for hidden data by detecting statistical anomalies in byte distributions.

### Education
Learn about information theory, randomness testing, and image structure through hands-on experimentation.

### Image Compression Analysis
Understand how different generation patterns affect compressibility by analyzing entropy.

## Metrics Interpretation

### Entropy
- **0-8 bits**: Information density
- **> 7.9**: Excellent randomness (cryptographic quality)
- **6.5-7.5**: Typical natural images
- **< 5.0**: Highly structured/predictable

### Chi-Square Test
- **p-value > 0.05**: Distribution appears uniform (good)
- **p-value < 0.05**: Non-uniform distribution detected (potential bias)

### Correlation
- **|r| < 0.1**: Excellent (negligible correlation)
- **|r| > 0.3**: Moderate correlation (patterns present)
- **|r| > 0.7**: Strong correlation (highly predictable)

### Monte Carlo π
- **Error < 1%**: Excellent 2D randomness
- **Error < 5%**: Acceptable randomness
- **Error > 10%**: Poor randomness (biased)

## Project Structure

```
pixel-lab/
├── src/pixel_lab/      # Core library
│   ├── generator.py    # ImageGenerator class
│   └── metrics.py      # ImageMetrics class
├── examples/           # Usage examples
├── tests/             # Unit tests
├── scripts/           # CLI tools
└── docs/              # Documentation
```

## Documentation (Coming soon)

- [Generator Guide](docs/generator_guide.md) - Complete ImageGenerator documentation
- [Metrics Guide](docs/metrics_guide.md) - Complete ImageMetrics documentation
- [Theory](docs/theory.md) - Cryptanalysis background and theory

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

GPL-3.0 License - See LICENSE file for details

## Citation

If you use Pixel Lab in your research, please cite:

```bibtex
@software{pixel_lab,
  title = {Pixel Lab: Pixel-Level Image Generation and Cryptanalytic Analysis},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/pixel-lab}
}
```

## Acknowledgments

Built with NumPy, Pillow, SciPy, and Matplotlib.

## Contact

For questions or feedback, please open an issue on GitHub.
