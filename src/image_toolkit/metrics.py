"""
Image Metrics Analyzer
A tool for computing cryptanalytic metrics on images to assess randomness and
statistical properties of pixel data.

Design decisions:
1. Focus on foundational metrics: entropy, frequency, chi-square, correlation
2. Support per-channel analysis (R, G, B) or combined analysis (ALL)
3. Use NumPy for efficient computation
4. Provide both programmatic API and CLI interface
5. Return detailed results with statistical interpretation

Image Metrics Analyzer - Usage Examples

PROGRAMMATIC USAGE:
-------------------
from image_toolkit import ImageMetrics

# Load and analyze an image
metrics = ImageMetrics("generated_image.png")

# Compute individual metrics
entropy = metrics.entropy(ImageMetrics.RGB.ALL)
chi2 = metrics.chi_square(ImageMetrics.RGB.RED)
corr = metrics.correlation(ImageMetrics.RGB.ALL, 'horizontal')
mad = metrics.mean_absolute_deviation(ImageMetrics.RGB.ALL)
mc_pi = metrics.monte_carlo_pi(ImageMetrics.RGB.ALL)

print(f"Entropy: {entropy:.4f} bits")
print(f"Chi-square p-value: {chi2['p_value']:.6f}")
print(f"Correlation: {corr:+.4f}")
print(f"MAD: {mad['mad_percentage']:.2f}%")
print(f"π estimate: {mc_pi['pi_estimate']:.6f}")

# Test randomness of generated image
good_random = (
    entropy > 7.9 and
    chi2['p_value'] > 0.05 and
    abs(corr) < 0.1 and
    mad['mad_percentage'] < 10.0 and
    mc_pi['error_percentage'] < 5.0
)
if good_random:
    print("✓ Image passes randomness tests")

# Get comprehensive analysis
analysis = metrics.analyze_all(ImageMetrics.RGB.ALL)

# Print formatted summary
print(metrics.summary(ImageMetrics.RGB.GREEN, verbose=True))

# Generate visualizations
metrics.plot_frequency_distribution(ImageMetrics.RGB.ALL)
metrics.plot_correlation_heatmap(ImageMetrics.RGB.ALL)
metrics.plot_monte_carlo_visualization(ImageMetrics.RGB.ALL)

# Or generate all plots at once
metrics.plot_all(ImageMetrics.RGB.ALL, save_dir='./output')

TESTING YOUR IMAGE GENERATOR:
-----------------------------
# Generate test image
from image_toolkit import ImageGenerator
import numpy as np

gen = ImageGenerator(512, 512)
# Fill with PRNG
for i in range(gen.byte_count):
    gen.set_byte(i, np.random.randint(0, 256))
gen.save("random_test.png")

# Analyze and visualize
metrics = ImageMetrics("random_test.png")
print(metrics.summary(ImageMetrics.RGB.ALL, verbose=True))
metrics.plot_all(ImageMetrics.RGB.ALL, save_dir='./test_results')

# Expected results for good PRNG:
# - Entropy: > 7.9 bits
# - Chi-square p-value: > 0.05
# - Correlations: < 0.1
# - MAD: < 10%
# - Monte Carlo π error: < 5%
"""

# NumPy: Core numerical computing - efficient array operations and statistical functions
import numpy as np

# PIL (Pillow): Image I/O - loading images in various formats
from PIL import Image

# Enum: Type-safe channel selection - prevents invalid channel specifications
from enum import Enum

# typing: Type hints for better code documentation and IDE support
from typing import Dict, Tuple, Optional

# scipy.stats: Statistical functions - chi-square test with p-value calculation
from scipy import stats

# matplotlib.pyplot: Plotting and visualization - histograms, correlation heatmaps
import matplotlib.pyplot as plt


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
    # Metric 3: Chi-Square Test
    # ========================================================================

    def chi_square(self, channel: RGB = RGB.ALL) -> Dict[str, float]:
        """
        Perform chi-square test for uniformity of byte distribution.

        Args:
            channel: Which channel(s) to analyze

        Returns:
            Dictionary containing:
            - 'statistic': Chi-square test statistic
            - 'p_value': Probability that observed distribution is random
            - 'dof': Degrees of freedom (255 for byte data)

        Chi-square formula: χ² = Σ((observed - expected)² / expected)

        Interpretation of p-value:
        - p > 0.05: Distribution appears uniform (passes randomness test)
        - p ≤ 0.05: Distribution is non-uniform (fails randomness test)
        - p < 0.01: Strong evidence of non-randomness

        Design decisions:
        1. Use scipy.stats.chisquare for accurate p-value calculation
        2. Expected frequency assumes perfect uniform distribution
        3. Return both statistic and p-value for flexibility in interpretation

        Use case: Testing if a PRNG produces uniform output, detecting bias
        in image generation algorithms.
        """
        observed = self.byte_frequency(channel)
        total = observed.sum()

        # Expected frequency for uniform distribution
        # Each of 256 values should appear total/256 times
        expected = np.full(256, total / 256.0)

        # Perform chi-square test
        # Design note: scipy automatically computes p-value from chi-square
        # distribution with appropriate degrees of freedom
        chi2_stat, p_value = stats.chisquare(observed, expected)

        return {
            'statistic': float(chi2_stat),
            'p_value': float(p_value),
            'dof': 255  # Degrees of freedom = num_categories - 1
        }

    # ========================================================================
    # Metric 4: Correlation Coefficient
    # ========================================================================

    def correlation(self, channel: RGB = RGB.ALL,
                   direction: str = 'horizontal', lag: int = 1) -> float:
        """
        Compute correlation coefficient between adjacent bytes.

        Args:
            channel: Which channel(s) to analyze
            direction: 'horizontal', 'vertical', or 'diagonal'
            lag: Distance between compared elements (default: 1 for adjacent)

        Returns:
            Pearson correlation coefficient (-1 to 1)

        Correlation interpretation:
        - 0.0: No correlation (ideal for randomness)
        - Close to 0 (|r| < 0.1): Very weak correlation (good)
        - |r| > 0.3: Moderate correlation (patterns present)
        - |r| > 0.7: Strong correlation (highly structured)

        Design decisions:
        1. Support multiple directions for comprehensive analysis
        2. Use Pearson correlation (measures linear relationship)
        3. Default lag=1 tests immediately adjacent elements
        4. For 'horizontal', compare sequential bytes in flattened array
        5. For 'vertical'/'diagonal', reshape to 2D and compute pixel correlations

        Use case: Detecting sequential patterns in PRNG output, analyzing
        spatial correlation in generated images.
        """
        data = self._get_channel_data(channel)

        if direction == 'horizontal':
            # Horizontal: correlation between consecutive bytes in flat array
            # Design note: This is most relevant for byte-sequence PRNGs
            if len(data) <= lag:
                raise ValueError("Lag too large for data size")

            # Split into pairs: [x0, x1, x2, ...] -> [x0, x1, ...] and [x1, x2, ...]
            x = data[:-lag].astype(np.float64)
            y = data[lag:].astype(np.float64)

        elif direction == 'vertical':
            # Vertical: correlation between vertically adjacent pixels
            # Need to work with 2D pixel data
            channel_2d = self._get_channel_2d(channel)

            if channel_2d.shape[0] <= lag:
                raise ValueError("Lag too large for image height")

            # Compare rows: pixels[row] with pixels[row+lag]
            x = channel_2d[:-lag, :].flatten().astype(np.float64)
            y = channel_2d[lag:, :].flatten().astype(np.float64)

        elif direction == 'diagonal':
            # Diagonal: correlation between diagonally adjacent pixels
            channel_2d = self._get_channel_2d(channel)

            if channel_2d.shape[0] <= lag or channel_2d.shape[1] <= lag:
                raise ValueError("Lag too large for image dimensions")

            # Compare diagonals: pixels[row, col] with pixels[row+lag, col+lag]
            x = channel_2d[:-lag, :-lag].flatten().astype(np.float64)
            y = channel_2d[lag:, lag:].flatten().astype(np.float64)

        else:
            raise ValueError(f"Invalid direction: {direction}. Use 'horizontal', 'vertical', or 'diagonal'")

        # Compute Pearson correlation coefficient
        # Design note: np.corrcoef returns a 2x2 correlation matrix,
        # we extract the off-diagonal element which is the correlation
        # between x and y
        if len(x) == 0 or len(y) == 0:
            return 0.0

        correlation_matrix = np.corrcoef(x, y)
        correlation_coef = correlation_matrix[0, 1]

        # Handle NaN case (occurs if variance is zero)
        if np.isnan(correlation_coef):
            return 0.0

        return float(correlation_coef)

    def _get_channel_2d(self, channel: RGB) -> np.ndarray:
        """
        Extract channel data as 2D array (for vertical/diagonal correlation).

        Args:
            channel: Which channel(s) to extract

        Returns:
            2D NumPy array (height x width) or (height x width*3) for ALL

        Design decision: Separate helper method to avoid duplicating
        2D extraction logic across correlation directions.
        """
        if channel == ImageMetrics.RGB.ALL:
            # For ALL channels, treat as grayscale-like 2D array
            # by averaging RGB values or flattening horizontally
            # Design choice: flatten RGB across width dimension
            return self.image.reshape(self.height, self.width * 3)

        elif channel == ImageMetrics.RGB.RED:
            return self.image[:, :, 0]

        elif channel == ImageMetrics.RGB.GREEN:
            return self.image[:, :, 1]

        elif channel == ImageMetrics.RGB.BLUE:
            return self.image[:, :, 2]

        else:
            raise ValueError(f"Invalid channel: {channel}")

    # ========================================================================
    # Metric 5: Mean Absolute Deviation from Uniform Distribution
    # ========================================================================

    def mean_absolute_deviation(self, channel: RGB = RGB.ALL) -> Dict[str, float]:
        """
        Compute Mean Absolute Deviation (MAD) from expected uniform distribution.

        Args:
            channel: Which channel(s) to analyze

        Returns:
            Dictionary containing:
            - 'mad': Mean absolute deviation value
            - 'mad_percentage': MAD as percentage of expected frequency
            - 'expected_frequency': Expected count per byte value (uniform)

        MAD formula: MAD = (1/256) × Σ|observed(i) - expected|

        Interpretation:
        - 0.0: Perfect uniform distribution
        - Lower values: Better uniformity
        - Higher values: More deviation from uniformity
        - As percentage: < 5% is excellent, < 10% is good, > 20% is poor

        Design decisions:
        1. Use absolute deviation rather than squared (unlike chi-square)
        2. Normalize by number of bins (256) for interpretability
        3. Provide percentage for intuitive understanding
        4. Complements chi-square by giving a different perspective on uniformity

        Use case: Simple, interpretable measure of distribution uniformity.
        Unlike chi-square, MAD is linear and easier to understand intuitively.
        """
        observed = self.byte_frequency(channel)
        total = observed.sum()

        # Expected frequency for uniform distribution
        expected = total / 256.0

        # Compute mean absolute deviation
        # Design note: Sum absolute differences, then divide by number of bins
        absolute_deviations = np.abs(observed - expected)
        mad = np.mean(absolute_deviations)

        # Express as percentage of expected frequency for interpretability
        mad_percentage = (mad / expected * 100.0) if expected > 0 else 0.0

        return {
            'mad': float(mad),
            'mad_percentage': float(mad_percentage),
            'expected_frequency': float(expected)
        }

    # ========================================================================
    # Metric 6: Monte Carlo Pi Estimation
    # ========================================================================

    def monte_carlo_pi(self, channel: RGB = RGB.ALL) -> Dict[str, float]:
        """
        Estimate π using Monte Carlo method as a randomness test.

        Args:
            channel: Which channel(s) to analyze

        Returns:
            Dictionary containing:
            - 'pi_estimate': Estimated value of π
            - 'error': Absolute error from true π
            - 'error_percentage': Error as percentage
            - 'points_used': Number of points used in estimation

        Monte Carlo π estimation method:
        1. Treat pairs of consecutive bytes as (x, y) coordinates in [0, 255]²
        2. Check if point falls inside quarter circle: x² + y² ≤ radius²
        3. Estimate π ≈ 4 × (points_inside / total_points)

        Interpretation:
        - True π ≈ 3.14159265359
        - Error < 1%: Excellent randomness
        - Error < 3%: Good randomness
        - Error < 5%: Acceptable randomness
        - Error > 10%: Poor randomness (biased distribution)

        Design decisions:
        1. Use consecutive byte pairs as coordinates (tests 2D randomness)
        2. Normalize to unit square for proper geometric interpretation
        3. Require even number of bytes (pairs of coordinates)
        4. Return both absolute and percentage error for easy interpretation
        5. This tests 2D spatial randomness, unlike 1D metrics (entropy, chi-square)

        Use case: Classic test that combines multiple bytes into geometric pattern.
        Sensitive to 2D correlations that other metrics might miss. A biased PRNG
        might pass chi-square but fail Monte Carlo π estimation.

        Note: Requires at least 2 bytes (1 point). More bytes = more accurate.
        """
        data = self._get_channel_data(channel)

        # Need even number of bytes to form coordinate pairs
        if len(data) < 2:
            raise ValueError("Need at least 2 bytes for Monte Carlo π estimation")

        # Use only even number of bytes (discard last byte if odd)
        num_pairs = len(data) // 2

        # Extract x and y coordinates from consecutive byte pairs
        # Design note: First byte of pair = x, second byte = y
        x_coords = data[0:num_pairs*2:2].astype(np.float64)
        y_coords = data[1:num_pairs*2:2].astype(np.float64)

        # Normalize coordinates to [0, 1] range
        # Design note: Divide by 255 (max byte value) to get unit square
        x_normalized = x_coords / 255.0
        y_normalized = y_coords / 255.0

        # Count points inside quarter circle (radius = 1, centered at origin)
        # Quarter circle equation: x² + y² ≤ 1
        distances_squared = x_normalized**2 + y_normalized**2
        points_inside = np.sum(distances_squared <= 1.0)

        # Estimate π using Monte Carlo formula
        # Area of quarter circle = π/4, so π ≈ 4 × (inside/total)
        pi_estimate = 4.0 * points_inside / num_pairs

        # Calculate error from true π
        true_pi = np.pi
        error = abs(pi_estimate - true_pi)
        error_percentage = (error / true_pi) * 100.0

        return {
            'pi_estimate': float(pi_estimate),
            'error': float(error),
            'error_percentage': float(error_percentage),
            'points_used': int(num_pairs),
            'true_pi': float(true_pi)
        }

    # ========================================================================
    # Visualization Methods
    # ========================================================================

    def plot_frequency_distribution(self, channel: RGB = RGB.ALL,
                                   save_path: Optional[str] = None) -> None:
        """
        Plot histogram of byte frequency distribution.

        Args:
            channel: Which channel(s) to analyze
            save_path: If provided, save plot to this path instead of displaying

        Design decisions:
        1. Show both observed distribution (bars) and expected uniform (line)
        2. Use different colors for different channels for clarity
        3. Include statistical info in title
        4. Optional save to file for reports/documentation

        Use case: Visual inspection of distribution uniformity, detecting outliers.
        """
        freq = self.byte_frequency(channel)
        expected = freq.sum() / 256.0

        # Set up the plot
        plt.figure(figsize=(12, 6))

        # Choose color based on channel
        color_map = {
            'ALL': 'gray',
            'RED': 'red',
            'GREEN': 'green',
            'BLUE': 'blue'
        }
        color = color_map.get(channel.name, 'gray')

        # Plot frequency distribution
        plt.bar(range(256), freq, color=color, alpha=0.7,
                label='Observed Frequency')

        # Plot expected uniform distribution line
        plt.axhline(y=expected, color='black', linestyle='--',
                   linewidth=2, label=f'Expected (Uniform): {expected:.2f}')

        # Add labels and title
        entropy = self.entropy(channel)
        chi2 = self.chi_square(channel)
        plt.xlabel('Byte Value (0-255)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Byte Frequency Distribution - {channel.name} Channel\n'
                 f'Entropy: {entropy:.4f} bits | '
                 f'Chi-Square p-value: {chi2["p_value"]:.6f}',
                 fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_correlation_heatmap(self, channel: RGB = RGB.ALL,
                                save_path: Optional[str] = None) -> None:
        """
        Plot 2D correlation heatmap between adjacent pixels.

        Args:
            channel: Which channel(s) to analyze
            save_path: If provided, save plot to this path instead of displaying

        Design decisions:
        1. Create scatter plot of (pixel_value, next_pixel_value) pairs
        2. Use 2D histogram (heatmap) to show density
        3. Perfect randomness shows uniform distribution across grid
        4. Patterns indicate correlation/predictability

        Use case: Visualizing sequential correlation, detecting patterns in PRNG output.
        A random sequence appears as uniform "noise", while patterns show structure.
        """
        data = self._get_channel_data(channel)

        if len(data) < 2:
            print("Need at least 2 bytes for correlation heatmap")
            return

        # Create pairs: (current_byte, next_byte)
        x = data[:-1]
        y = data[1:]

        # Set up the plot
        plt.figure(figsize=(10, 10))

        # Create 2D histogram heatmap
        # Design note: 256x256 bins for full byte value resolution
        plt.hist2d(x, y, bins=256, range=[[0, 255], [0, 255]],
                  cmap='hot', cmin=1)

        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label('Frequency', fontsize=12)

        # Add correlation coefficient to title
        corr = self.correlation(channel, 'horizontal')
        plt.xlabel('Current Byte Value', fontsize=12)
        plt.ylabel('Next Byte Value', fontsize=12)
        plt.title(f'Sequential Correlation Heatmap - {channel.name} Channel\n'
                 f'Correlation Coefficient: {corr:+.6f}',
                 fontsize=14)

        # Add diagonal line (perfect correlation reference)
        plt.plot([0, 255], [0, 255], 'c--', linewidth=2, alpha=0.5,
                label='Perfect Correlation')
        plt.legend()

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_monte_carlo_visualization(self, channel: RGB = RGB.ALL,
                                      max_points: int = 10000,
                                      save_path: Optional[str] = None) -> None:
        """
        Visualize Monte Carlo π estimation by plotting points.

        Args:
            channel: Which channel(s) to analyze
            max_points: Maximum number of points to plot (for performance)
            save_path: If provided, save plot to this path instead of displaying

        Design decisions:
        1. Plot points colored by inside/outside quarter circle
        2. Limit points for visualization clarity and performance
        3. Show estimated π value in title
        4. Draw quarter circle boundary for reference

        Use case: Visual verification of 2D randomness, educational demonstration
        of Monte Carlo method.
        """
        data = self._get_channel_data(channel)

        if len(data) < 2:
            print("Need at least 2 bytes for Monte Carlo visualization")
            return

        # Limit number of points for visualization
        num_pairs = min(len(data) // 2, max_points)

        # Extract and normalize coordinates
        x = data[0:num_pairs*2:2].astype(np.float64) / 255.0
        y = data[1:num_pairs*2:2].astype(np.float64) / 255.0

        # Determine which points are inside quarter circle
        distances_squared = x**2 + y**2
        inside = distances_squared <= 1.0

        # Compute π estimate
        pi_result = self.monte_carlo_pi(channel)

        # Set up the plot
        plt.figure(figsize=(10, 10))

        # Plot points inside and outside circle with different colors
        plt.scatter(x[inside], y[inside], c='blue', s=1, alpha=0.5,
                   label=f'Inside Circle ({np.sum(inside)} points)')
        plt.scatter(x[~inside], y[~inside], c='red', s=1, alpha=0.5,
                   label=f'Outside Circle ({np.sum(~inside)} points)')

        # Draw quarter circle boundary
        theta = np.linspace(0, np.pi/2, 100)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        plt.plot(circle_x, circle_y, 'black', linewidth=2, label='Quarter Circle')

        # Format plot
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('X (Normalized)', fontsize=12)
        plt.ylabel('Y (Normalized)', fontsize=12)
        plt.title(f'Monte Carlo π Estimation - {channel.name} Channel\n'
                 f'π Estimate: {pi_result["pi_estimate"]:.6f} | '
                 f'True π: {pi_result["true_pi"]:.6f} | '
                 f'Error: {pi_result["error_percentage"]:.2f}%',
                 fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().set_aspect('equal', adjustable='box')

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_all(self, channel: RGB = RGB.ALL,
                save_dir: Optional[str] = None) -> None:
        """
        Generate all visualization plots at once.

        Args:
            channel: Which channel(s) to analyze
            save_dir: If provided, save all plots to this directory

        Design decision: Convenience method to generate comprehensive
        visual report with a single call.
        """
        import os

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            freq_path = os.path.join(save_dir, f'frequency_{channel.name.lower()}.png')
            corr_path = os.path.join(save_dir, f'correlation_{channel.name.lower()}.png')
            monte_path = os.path.join(save_dir, f'monte_carlo_{channel.name.lower()}.png')
        else:
            freq_path = None
            corr_path = None
            monte_path = None

        print(f"\nGenerating visualizations for {channel.name} channel...")
        print("1/3: Frequency distribution...")
        self.plot_frequency_distribution(channel, freq_path)

        print("2/3: Correlation heatmap...")
        self.plot_correlation_heatmap(channel, corr_path)

        print("3/3: Monte Carlo visualization...")
        self.plot_monte_carlo_visualization(channel, save_path=monte_path)

        print("✓ All visualizations complete!")

    # ========================================================================
    # Analysis and Reporting (Updated)
    # ========================================================================

    def analyze_all(self, channel: RGB = RGB.ALL) -> Dict:
        """
        Compute all available metrics for the specified channel.

        Args:
            channel: Which channel(s) to analyze

        Returns:
            Dictionary containing all metrics and statistical information

        Design decision: Provide a comprehensive analysis method that
        computes all metrics at once for convenience and reporting.
        """
        data = self._get_channel_data(channel)
        freq = self.byte_frequency(channel)
        chi2_result = self.chi_square(channel)
        mad_result = self.mean_absolute_deviation(channel)
        mc_pi_result = self.monte_carlo_pi(channel)

        return {
            'channel': channel.name,
            'entropy': self.entropy(channel),
            'chi_square': chi2_result,
            'correlation_horizontal': self.correlation(channel, 'horizontal'),
            'correlation_vertical': self.correlation(channel, 'vertical'),
            'correlation_diagonal': self.correlation(channel, 'diagonal'),
            'mean_absolute_deviation': mad_result,
            'monte_carlo_pi': mc_pi_result,
            'byte_frequency': freq,
            'unique_values': int(np.count_nonzero(freq)),
            'total_bytes': len(data),
            'mean': float(np.mean(data)),
            'std_dev': float(np.std(data)),
            'min': int(np.min(data)),
            'max': int(np.max(data))
        }

    def summary(self, channel: RGB = RGB.ALL, verbose: bool = False) -> str:
        """
        Generate a human-readable summary report.

        Args:
            channel: Which channel(s) to analyze
            verbose: If True, include detailed interpretation guidance

        Returns:
            Formatted string report

        Design decision: Separate summary formatting from computation
        to allow flexible output formats (JSON, text, etc.)
        """
        analysis = self.analyze_all(channel)

        # Interpret entropy
        entropy = analysis['entropy']
        if entropy > 7.9:
            entropy_quality = "Excellent (cryptographically random)"
        elif entropy > 7.5:
            entropy_quality = "Good (high randomness)"
        elif entropy > 7.0:
            entropy_quality = "Moderate (typical for natural images)"
        elif entropy > 5.0:
            entropy_quality = "Low (significant patterns present)"
        else:
            entropy_quality = "Very Low (highly structured/uniform)"

        # Interpret chi-square p-value
        p_value = analysis['chi_square']['p_value']
        if p_value > 0.05:
            chi2_quality = "PASS (appears uniform)"
        elif p_value > 0.01:
            chi2_quality = "MARGINAL (slight non-uniformity)"
        else:
            chi2_quality = "FAIL (significantly non-uniform)"

        # Interpret correlations
        corr_h = abs(analysis['correlation_horizontal'])
        corr_v = abs(analysis['correlation_vertical'])
        corr_d = abs(analysis['correlation_diagonal'])

        def interpret_correlation(corr):
            if corr < 0.1:
                return "Excellent (negligible correlation)"
            elif corr < 0.3:
                return "Good (weak correlation)"
            elif corr < 0.5:
                return "Moderate (noticeable patterns)"
            elif corr < 0.7:
                return "Poor (strong correlation)"
            else:
                return "Very Poor (highly correlated)"

        # Interpret MAD
        mad_pct = analysis['mean_absolute_deviation']['mad_percentage']
        if mad_pct < 5.0:
            mad_quality = "Excellent (very uniform)"
        elif mad_pct < 10.0:
            mad_quality = "Good (reasonably uniform)"
        elif mad_pct < 20.0:
            mad_quality = "Moderate (some deviation)"
        else:
            mad_quality = "Poor (highly non-uniform)"

        # Interpret Monte Carlo π
        mc_error_pct = analysis['monte_carlo_pi']['error_percentage']
        if mc_error_pct < 1.0:
            mc_quality = "Excellent (highly random)"
        elif mc_error_pct < 3.0:
            mc_quality = "Good (random)"
        elif mc_error_pct < 5.0:
            mc_quality = "Acceptable (mostly random)"
        elif mc_error_pct < 10.0:
            mc_quality = "Marginal (some bias)"
        else:
            mc_quality = "Poor (significant bias)"

        report = f"""
{'='*70}
Image Metrics Analysis
{'='*70}
File: {self.filename}
Dimensions: {self.width} × {self.height} pixels
Channel: {analysis['channel']}

SHANNON ENTROPY
  Value: {entropy:.6f} bits
  Quality: {entropy_quality}
  Theoretical Maximum: 8.000000 bits

CHI-SQUARE TEST (Uniformity)
  Statistic: {analysis['chi_square']['statistic']:.2f}
  P-Value: {p_value:.6f}
  Result: {chi2_quality}
  Degrees of Freedom: {analysis['chi_square']['dof']}

MEAN ABSOLUTE DEVIATION
  MAD Value: {analysis['mean_absolute_deviation']['mad']:.2f}
  MAD Percentage: {mad_pct:.2f}%
  Quality: {mad_quality}
  Expected Frequency: {analysis['mean_absolute_deviation']['expected_frequency']:.2f}

MONTE CARLO π ESTIMATION
  Estimated π: {analysis['monte_carlo_pi']['pi_estimate']:.6f}
  True π: {analysis['monte_carlo_pi']['true_pi']:.6f}
  Error: {analysis['monte_carlo_pi']['error']:.6f} ({mc_error_pct:.2f}%)
  Quality: {mc_quality}
  Points Used: {analysis['monte_carlo_pi']['points_used']:,}

CORRELATION ANALYSIS
  Horizontal: {analysis['correlation_horizontal']:+.6f} - {interpret_correlation(corr_h)}
  Vertical:   {analysis['correlation_vertical']:+.6f} - {interpret_correlation(corr_v)}
  Diagonal:   {analysis['correlation_diagonal']:+.6f} - {interpret_correlation(corr_d)}

BYTE STATISTICS
  Total Bytes: {analysis['total_bytes']:,}
  Unique Values: {analysis['unique_values']}/256 ({100*analysis['unique_values']/256:.1f}%)
  Mean: {analysis['mean']:.2f}
  Std Dev: {analysis['std_dev']:.2f}
  Range: [{analysis['min']}, {analysis['max']}]

DISTRIBUTION ANALYSIS
  Most Common Value: {np.argmax(analysis['byte_frequency'])} (appears {np.max(analysis['byte_frequency']):,} times)
  Expected per Value (uniform): {analysis['total_bytes']/256:.2f}
{'='*70}
"""

        if verbose:
            report += """
INTERPRETATION GUIDE:

  Entropy: Measures information density
    - Higher is better for randomness (target: > 7.9)
    - Natural images typically: 6.5-7.5
    - Compressed/encrypted data: > 7.9

  Chi-Square: Tests if distribution is uniform
    - P-value > 0.05: Cannot reject uniformity (good)
    - P-value < 0.05: Distribution is biased (bad)

  Mean Absolute Deviation: Average deviation from uniform
    - < 5%: Excellent uniformity
    - < 10%: Good uniformity
    - > 20%: Poor uniformity

  Monte Carlo π: 2D randomness test
    - Error < 1%: Excellent 2D randomness
    - Error < 5%: Acceptable 2D randomness
    - Error > 10%: Poor 2D randomness (biased)

  Correlation: Measures predictability from neighbors
    - Target: Close to 0.0 (no predictable patterns)
    - |r| < 0.1: Good randomness
    - |r| > 0.3: Detectable patterns present
{'='*70}
"""

        return report
