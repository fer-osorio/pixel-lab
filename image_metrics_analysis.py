"""
Image Metrics Analyzer
A tool for computing cryptanalytic metrics on images to assess randomness and
statistical properties of pixel data.

Design decisions:
1. Focus on Shannon entropy and byte frequency distribution as foundational metrics
2. Support per-channel analysis (R, G, B) or combined analysis (ALL)
3. Use NumPy for efficient computation
4. Provide both programmatic API and CLI interface
"""

import numpy as np
from PIL import Image
from enum import Enum
from typing import Dict, Tuple
import argparse
import sys


class ImageMetrics:
    """
    Analyze statistical and cryptanalytic properties of images.

    Primary use cases:
    - Validating PRNG-based image generation
    - Detecting patterns or bias in generated images
    - Assessing randomness for steganography or compression
    """

    class RGB(Enum):
        """Enum for specifying which color channel(s) to analyze."""
        ALL = 0    # Analyze all bytes (R, G, B interleaved)
        RED = 1    # Analyze only red channel
        GREEN = 2  # Analyze only green channel
        BLUE = 3   # Analyze only blue channel

    def __init__(self, filename: str):
        """
        Open an image file and load it into a NumPy array.

        Args:
            filename: Path to image file (any PIL-supported format)

        Design decision: Convert to RGB mode to ensure consistent 3-channel
        structure, even if the input is grayscale or RGBA.
        """
        try:
            img = Image.open(filename)
            # Convert to RGB to ensure 3 channels (handles RGBA, grayscale, etc.)
            img_rgb = img.convert('RGB')
            self.image = np.array(img_rgb, dtype=np.uint8)
            self._filename = filename

        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {filename}")
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")

        # Validate image was loaded correctly
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError("Image must have 3 color channels (RGB)")

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self.image.shape[1]

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self.image.shape[0]

    @property
    def filename(self) -> str:
        """Original filename."""
        return self._filename

    # ========================================================================
    # Internal Helper Methods
    # ========================================================================

    def _get_channel_data(self, channel: RGB) -> np.ndarray:
        """
        Extract byte data for the specified channel.

        Args:
            channel: Which channel(s) to extract

        Returns:
            1D NumPy array of bytes (uint8)

        Design decision: Return a flattened array of bytes for easy
        statistical analysis. The order doesn't matter for most metrics.
        """
        if channel == ImageMetrics.RGB.ALL:
            # Return all bytes in row-major order: R0, G0, B0, R1, G1, B1, ...
            return self.image.reshape(-1)

        elif channel == ImageMetrics.RGB.RED:
            # Extract only red channel (index 0)
            return self.image[:, :, 0].reshape(-1)

        elif channel == ImageMetrics.RGB.GREEN:
            # Extract only green channel (index 1)
            return self.image[:, :, 1].reshape(-1)

        elif channel == ImageMetrics.RGB.BLUE:
            # Extract only blue channel (index 2)
            return self.image[:, :, 2].reshape(-1)

        else:
            raise ValueError(f"Invalid channel: {channel}")

    # ========================================================================
    # Metric 1: Byte Frequency Distribution
    # ========================================================================

    def byte_frequency(self, channel: RGB = RGB.ALL) -> np.ndarray:
        """
        Compute the frequency distribution of byte values (0-255).

        Args:
            channel: Which channel(s) to analyze

        Returns:
            NumPy array of length 256, where index i contains the count of
            bytes with value i

        Design decision: Return raw counts rather than probabilities to
        preserve precision and allow caller to normalize if needed.

        Use case: Visualizing distribution, detecting bias, computing chi-square.
        """
        data = self._get_channel_data(channel)

        # Use NumPy's bincount for efficient counting
        # bins parameter ensures we get all 256 possible byte values
        # even if some don't appear in the data
        frequency = np.bincount(data, minlength=256)

        return frequency

    def byte_frequency_normalized(self, channel: RGB = RGB.ALL) -> np.ndarray:
        """
        Compute the normalized frequency distribution (probabilities).

        Args:
            channel: Which channel(s) to analyze

        Returns:
            NumPy array of length 256, where index i contains the probability
            of bytes with value i. Sum of all elements equals 1.0.

        Design decision: Provide this as a separate method since normalized
        probabilities are needed for entropy calculation and other metrics.
        """
        frequency = self.byte_frequency(channel)
        total = frequency.sum()

        # Avoid division by zero (shouldn't happen with valid images)
        if total == 0:
            return np.zeros(256)

        return frequency.astype(np.float64) / total

    # ========================================================================
    # Metric 2: Shannon Entropy
    # ========================================================================

    def entropy(self, channel: RGB = RGB.ALL) -> float:
        """
        Compute Shannon entropy of the byte distribution.

        Args:
            channel: Which channel(s) to analyze

        Returns:
            Entropy in bits (range: 0 to 8 for byte data)

        Shannon entropy formula: H = -Σ(p(i) × log₂(p(i)))
        where p(i) is the probability of byte value i.

        Interpretation:
        - 0.0: All bytes have the same value (no randomness)
        - 8.0: Perfect uniform distribution (maximum randomness for bytes)
        - Typical natural images: 6.5 - 7.5
        - Good PRNG output: > 7.9

        Design decision: Use log base 2 to measure entropy in bits, which
        is standard in information theory and cryptanalysis.
        """
        probabilities = self.byte_frequency_normalized(channel)

        # Filter out zero probabilities to avoid log(0)
        # Design note: 0 × log(0) is defined as 0 in information theory
        nonzero_probs = probabilities[probabilities > 0]

        # Compute Shannon entropy: -Σ(p × log₂(p))
        entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs))

        return float(entropy)

    # ========================================================================
    # Analysis and Reporting
    # ========================================================================

    def analyze_basic(self, channel: RGB = RGB.ALL) -> Dict:
        """
        Compute basic metrics for the specified channel.

        Args:
            channel: Which channel(s) to analyze

        Returns:
            Dictionary containing:
            - 'entropy': Shannon entropy
            - 'byte_frequency': Frequency distribution array
            - 'unique_values': Number of unique byte values present
            - 'total_bytes': Total number of bytes analyzed

        Design decision: Return a dictionary for easy extension with
        additional metrics later.
        """
        data = self._get_channel_data(channel)
        freq = self.byte_frequency(channel)

        return {
            'channel': channel.name,
            'entropy': self.entropy(channel),
            'byte_frequency': freq,
            'unique_values': int(np.count_nonzero(freq)),
            'total_bytes': len(data),
            'mean': float(np.mean(data)),
            'std_dev': float(np.std(data)),
            'min': int(np.min(data)),
            'max': int(np.max(data))
        }

    def summary(self, channel: RGB = RGB.ALL) -> str:
        """
        Generate a human-readable summary report.

        Args:
            channel: Which channel(s) to analyze

        Returns:
            Formatted string report

        Design decision: Separate summary formatting from computation
        to allow flexible output formats (JSON, text, etc.)
        """
        analysis = self.analyze_basic(channel)

        # Determine randomness quality based on entropy
        entropy = analysis['entropy']
        if entropy > 7.9:
            quality = "Excellent (cryptographically random)"
        elif entropy > 7.5:
            quality = "Good (high randomness)"
        elif entropy > 7.0:
            quality = "Moderate (typical for natural images)"
        elif entropy > 5.0:
            quality = "Low (significant patterns present)"
        else:
            quality = "Very Low (highly structured/uniform)"

        report = f"""
{'='*60}
Image Metrics Analysis
{'='*60}
File: {self.filename}
Dimensions: {self.width} × {self.height} pixels
Channel: {analysis['channel']}

SHANNON ENTROPY
  Value: {entropy:.4f} bits
  Quality: {quality}
  Theoretical Maximum: 8.000 bits

BYTE STATISTICS
  Total Bytes: {analysis['total_bytes']:,}
  Unique Values: {analysis['unique_values']}/256 ({100*analysis['unique_values']/256:.1f}%)
  Mean: {analysis['mean']:.2f}
  Std Dev: {analysis['std_dev']:.2f}
  Range: [{analysis['min']}, {analysis['max']}]

DISTRIBUTION ANALYSIS
  Most Common Value: {np.argmax(analysis['byte_frequency'])} (appears {np.max(analysis['byte_frequency'])} times)
  Least Common Value: {np.argmin(analysis['byte_frequency'][analysis['byte_frequency'] > 0]) if np.any(analysis['byte_frequency'] > 0) else 'N/A'}
{'='*60}
"""
        return report


# ============================================================================
# Command-Line Interface
# ============================================================================

def main():
    """
    CLI entry point for analyzing images from the command line.

    Design decision: Provide a simple CLI for quick analysis without
    writing Python code. Useful for batch processing or integration
    with other tools.
    """
    parser = argparse.ArgumentParser(
        description='Analyze statistical and cryptanalytic properties of images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.png
  %(prog)s image.png --channel red
  %(prog)s image.png --channel all --verbose

Channel options: all, red, green, blue
        """
    )

    parser.add_argument(
        'image',
        help='Path to image file'
    )

    parser.add_argument(
        '--channel', '-c',
        choices=['all', 'red', 'green', 'blue'],
        default='all',
        help='Color channel to analyze (default: all)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed frequency distribution'
    )

    args = parser.parse_args()

    # Map string argument to enum
    channel_map = {
        'all': ImageMetrics.RGB.ALL,
        'red': ImageMetrics.RGB.RED,
        'green': ImageMetrics.RGB.GREEN,
        'blue': ImageMetrics.RGB.BLUE
    }
    channel = channel_map[args.channel]

    try:
        # Load and analyze image
        metrics = ImageMetrics(args.image)

        # Print summary report
        print(metrics.summary(channel))

        # Optionally show detailed frequency distribution
        if args.verbose:
            freq = metrics.byte_frequency(channel)
            print("\nDETAILED FREQUENCY DISTRIBUTION")
            print("Value | Count    | Percentage")
            print("-" * 35)
            total = freq.sum()
            for value in range(256):
                if freq[value] > 0:
                    percentage = 100 * freq[value] / total
                    print(f"{value:3d}   | {freq[value]:8d} | {percentage:6.3f}%")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    # If command-line arguments provided, run CLI
    if len(sys.argv) > 1:
        main()
    else:
        # Otherwise, show usage examples
        print("""
Image Metrics Analyzer - Usage Examples

PROGRAMMATIC USAGE:
-------------------
from image_metrics import ImageMetrics

# Load and analyze an image
metrics = ImageMetrics("generated_image.png")

# Compute entropy for all channels
entropy = metrics.entropy(ImageMetrics.RGB.ALL)
print(f"Entropy: {entropy:.4f} bits")

# Analyze specific color channel
red_entropy = metrics.entropy(ImageMetrics.RGB.RED)
print(f"Red channel entropy: {red_entropy:.4f} bits")

# Get frequency distribution
freq = metrics.byte_frequency(ImageMetrics.RGB.ALL)
print(f"Most common byte value: {freq.argmax()}")

# Get comprehensive analysis
analysis = metrics.analyze_basic(ImageMetrics.RGB.ALL)
print(analysis)

# Print formatted summary
print(metrics.summary(ImageMetrics.RGB.GREEN))

COMMAND-LINE USAGE:
-------------------
python image_metrics.py <image_file> [options]

Options:
  --channel, -c  : Specify channel (all, red, green, blue)
  --verbose, -v  : Show detailed frequency distribution

Examples:
  python image_metrics.py image.png
  python image_metrics.py image.png --channel red
  python image_metrics.py image.png -c all -v

To see this help message, run without arguments.
To analyze an image, provide the image path as an argument.
        """)
