"""
Integration Tests for Pixel Lab

Tests complete workflows combining ImageGenerator and ImageMetrics.
These tests verify that components work together correctly end-to-end.

Tests organized by workflow:
1. Generation → Analysis pipeline (3 tests)
2. Recursive generation workflows (2 tests)
3. Full analysis with visualization (2 tests)
4. Round-trip consistency (1 test)

Total: 8 integration tests

Design philosophy:
- Test realistic user workflows
- Verify component interaction
- Allow longer execution time than unit tests
- File I/O is expected and tested
"""

# pathlib.Path: Type hints for paths
from pathlib import Path

# typing: Type hints for function signatures
from typing import Tuple

# numpy: Generate test data and validate results
import numpy as np
import pytest

# Import both classes for integration testing
from pixel_lab import ImageGenerator, ImageMetrics

# ============================================================================
# 1. Generation → Analysis Pipeline Tests
# ============================================================================


def test_generate_random_analyze_passes(tmp_path: Path) -> None:
    """
    Integration test: Generate random image → Analyze → Verify passes tests.

    Workflow:
    1. Generate 256x256 image with good PRNG
    2. Analyze all metrics
    3. Verify passes randomness tests

    Design decision: This is the primary use case - validating PRNG output.
    """
    image_path = tmp_path / "random_test.png"

    # Step 1: Generate random image
    gen = ImageGenerator(256, 256)
    rng = np.random.RandomState(1234)  # Good PRNG

    for i in range(gen.byte_count):
        gen.set_byte(i, rng.randint(0, 256))

    gen.save(str(image_path))

    # Step 2: Analyze
    metrics = ImageMetrics(str(image_path))
    analysis = metrics.analyze_all(ImageMetrics.RGB.ALL)

    # Step 3: Verify randomness quality
    assert analysis["entropy"] > 7.9, (
        f"PRNG should have high entropy, got {analysis['entropy']:.4f}"
    )

    assert analysis["chi_square"]["p_value"] > 0.05, (
        f"PRNG should pass chi-square, got p={analysis['chi_square']['p_value']:.6f}"
    )

    assert abs(analysis["correlation_horizontal"]) < 0.1, (
        f"PRNG should have low correlation, got {analysis['correlation_horizontal']:.4f}"
    )

    assert analysis["mean_absolute_deviation"]["mad_percentage"] < 15.0, (
        f"PRNG should have low MAD, got {analysis['mean_absolute_deviation']['mad_percentage']:.2f}%"
    )

    assert analysis["monte_carlo_pi"]["error_percentage"] < 10.0, (
        f"PRNG should estimate π well, got {analysis['monte_carlo_pi']['error_percentage']:.2f}% error"
    )


def test_generate_biased_analyze_fails(tmp_path: Path) -> None:
    """
    Integration test: Generate biased image → Analyze → Verify fails tests.

    Workflow:
    1. Generate image with obvious bias (alternating 0 and 255)
    2. Analyze all metrics
    3. Verify detects non-randomness

    Design decision: Verify tool can detect bad PRNGs/patterns.
    """
    image_path = tmp_path / "biased_test.png"

    # Step 1: Generate biased image (alternating pattern)
    gen = ImageGenerator(100, 100)

    for i in range(gen.byte_count):
        gen.set_byte(i, 0 if i % 2 == 0 else 255)

    gen.save(str(image_path))

    # Step 2: Analyze
    metrics = ImageMetrics(str(image_path))
    analysis = metrics.analyze_all(ImageMetrics.RGB.ALL)

    # Step 3: Verify detects bias
    assert analysis["entropy"] < 2.0, (
        f"Biased pattern should have low entropy, got {analysis['entropy']:.4f}"
    )

    assert analysis["chi_square"]["p_value"] < 0.05, (
        f"Biased pattern should fail chi-square, got p={analysis['chi_square']['p_value']:.6f}"
    )


def test_generate_save_load_verify_consistency(tmp_path: Path) -> None:
    """
    Integration test: Generate → Save → Load → Verify pixel consistency.

    Workflow:
    1. Generate image with known pixel values
    2. Save to file
    3. Load with ImageMetrics
    4. Verify pixel values preserved

    Design decision: Verify no data corruption in save/load cycle.
    """
    image_path = tmp_path / "consistency_test.png"

    # Step 1: Generate with known pattern
    gen = ImageGenerator(50, 50)

    # Create checkerboard pattern for easy verification
    for y in range(gen.height):
        for x in range(gen.width):
            if (x + y) % 2 == 0:
                gen.set_pixel(x, y, 255, 0, 0)  # Red
            else:
                gen.set_pixel(x, y, 0, 0, 255)  # Blue

    # Step 2: Save
    gen.save(str(image_path))

    # Step 3: Load
    metrics = ImageMetrics(str(image_path))

    # Step 4: Verify pattern preserved
    for y in range(50):
        for x in range(50):
            if (x + y) % 2 == 0:
                assert np.array_equal(metrics.image[y, x], [255, 0, 0]), (
                    f"Red pixel at ({x}, {y}) corrupted"
                )
            else:
                assert np.array_equal(metrics.image[y, x], [0, 0, 255]), (
                    f"Blue pixel at ({x}, {y}) corrupted"
                )


# ============================================================================
# 2. Recursive Generation Workflow Tests
# ============================================================================


def test_recursive_pixel_generation_creates_gradient(tmp_path: Path) -> None:
    """
    Integration test: Use recursive pixel generation → Verify creates gradient.

    Workflow:
    1. Define gradient function
    2. Apply recursively to all pixels
    3. Analyze correlation (should be high)

    Design decision: Verify recursive generation creates expected patterns.
    """
    image_path = tmp_path / "recursive_gradient.png"

    # Step 1: Define gradient function
    def gradient_func(x: int, y: int, img: np.ndarray) -> Tuple[int, int, int]:
        # Gradual increase based on position
        if x == 0:
            r = 0
        else:
            r = (img[y][x - 1][0] + 2) & 255
        if y == 0:
            g = 0
        else:
            g = (img[y - 1][x][1] + 2) & 255
        b = 128
        return (r, g, b)

    # Step 2: Generate using recursive function
    gen = ImageGenerator(128, 128)

    for y in range(gen.height):
        for x in range(gen.width):
            gen.set_pixel_recursive(x, y, gradient_func)

    gen.save(str(image_path))

    # Step 3: Analyze
    metrics = ImageMetrics(str(image_path))

    # Gradient should have high spatial correlation
    corr_h = metrics.correlation(ImageMetrics.RGB.RED, "horizontal")
    corr_v = metrics.correlation(ImageMetrics.RGB.GREEN, "vertical")

    assert abs(corr_h) == 1.0, (
        f"Gradient should have high horizontal red correlation, got {corr_h:.4f}"
    )
    assert abs(corr_v) == 1.0, (
        f"Gradient should have high vertical green correlation, got {corr_v:.4f}"
    )


def test_recursive_byte_generation_creates_sequence(tmp_path: Path) -> None:
    """
    Integration test: Use recursive byte generation → Verify creates sequence.

    Workflow:
    1. Define byte sequence function (Fibonacci-like)
    2. Apply recursively
    3. Verify pattern in output

    Design decision: Test byte-level recursive generation capability.
    """
    image_path = tmp_path / "recursive_sequence.png"

    # Step 1: Define sequence function
    def fibonacci_byte(idx: int, flat: np.ndarray) -> int:
        if idx < 2:
            return 1
        # Fibonacci-like: sum of two previous, mod 256
        return (int(flat[idx - 1]) + int(flat[idx - 2])) % 256

    # Step 2: Generate using recursive function
    gen = ImageGenerator(64, 64)  # 12,288 bytes

    for i in range(gen.byte_count):
        gen.set_byte_recursive(i, fibonacci_byte)

    gen.save(str(image_path))

    # Step 3: Verify sequence properties
    # Fibonacci-like sequences should have some pattern but not be constant
    metrics = ImageMetrics(str(image_path))

    entropy = metrics.entropy(ImageMetrics.RGB.ALL)
    unique_values = len(np.unique(metrics.image))

    # Should have moderate entropy (not random, not constant)
    assert 4.0 < entropy < 7.5, (
        f"Fibonacci sequence should have moderate entropy, got {entropy:.4f}"
    )

    # Should have multiple unique values
    assert unique_values > 10, (
        f"Fibonacci sequence should have variety, got {unique_values} unique values"
    )


# ============================================================================
# 3. Full Analysis with Visualization Tests
# ============================================================================


def test_complete_analysis_workflow(tmp_path: Path) -> None:
    """
    Integration test: Generate → Analyze → Visualize → Verify all outputs.

    Workflow:
    1. Generate random image
    2. Run complete analysis
    3. Generate all visualizations
    4. Verify all files created

    Design decision: Test the complete end-to-end user workflow.
    """
    image_path = tmp_path / "complete_test.png"
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()

    # Step 1: Generate
    gen = ImageGenerator(200, 200)
    rng = np.random.RandomState(99999)

    for i in range(gen.byte_count):
        gen.set_byte(i, rng.randint(0, 256))

    gen.save(str(image_path))

    # Step 2: Analyze
    metrics = ImageMetrics(str(image_path))
    analysis = metrics.analyze_all(ImageMetrics.RGB.ALL)
    summary = metrics.summary(ImageMetrics.RGB.ALL, verbose=True)

    # Verify analysis contains all expected metrics
    assert "entropy" in analysis
    assert "chi_square" in analysis
    assert "correlation_horizontal" in analysis
    assert "mean_absolute_deviation" in analysis
    assert "monte_carlo_pi" in analysis

    # Verify summary is substantial
    assert len(summary) > 500, "Summary should be comprehensive"
    assert "INTERPRETATION GUIDE" in summary

    # Step 3: Visualize
    metrics.plot_all(ImageMetrics.RGB.ALL, save_dir=str(plots_dir))

    # Step 4: Verify all plot files created
    expected_plots = ["frequency_all.png", "correlation_all.png", "monte_carlo_all.png"]

    for plot_name in expected_plots:
        plot_path = plots_dir / plot_name
        assert plot_path.exists(), f"Plot {plot_name} not created"
        assert plot_path.stat().st_size > 1000, f"Plot {plot_name} seems empty"


def test_multi_channel_analysis(tmp_path: Path) -> None:
    """
    Integration test: Analyze same image across all channels → Compare results.

    Workflow:
    1. Generate image with different characteristics per channel
    2. Analyze ALL, RED, GREEN, BLUE separately
    3. Verify results differ appropriately

    Design decision: Verify channel isolation works correctly.
    """
    image_path = tmp_path / "multi_channel.png"

    # Step 1: Generate with channel-specific patterns
    gen = ImageGenerator(100, 100)
    rng = np.random.RandomState(777)

    for y in range(gen.height):
        for x in range(gen.width):
            # Red: random
            r = rng.randint(0, 256)
            # Green: gradient
            g = min(255, x + y)
            # Blue: constant
            b = 128
            gen.set_pixel(x, y, r, g, b)

    gen.save(str(image_path))

    # Step 2: Analyze all channels
    metrics = ImageMetrics(str(image_path))

    entropy_all = metrics.entropy(ImageMetrics.RGB.ALL)
    entropy_red = metrics.entropy(ImageMetrics.RGB.RED)
    entropy_green = metrics.entropy(ImageMetrics.RGB.GREEN)
    entropy_blue = metrics.entropy(ImageMetrics.RGB.BLUE)

    # Step 3: Verify expected differences
    # RED should have high entropy (random)
    assert entropy_red > 7.5, (
        f"Random RED should have high entropy, got {entropy_red:.4f}"
    )

    # GREEN should have moderate entropy (gradient)
    assert 6.0 < entropy_green < 8.0, (
        f"Gradient GREEN should have moderate entropy, got {entropy_green:.4f}"
    )

    # BLUE should have low entropy (constant)
    assert entropy_blue < 0.1, (
        f"Constant BLUE should have near-zero entropy, got {entropy_blue:.4f}"
    )

    # ALL should be somewhere in between
    assert entropy_blue < entropy_all < entropy_red, (
        "ALL channel entropy should be between extremes"
    )


# ============================================================================
# 4. Round-Trip Consistency Test
# ============================================================================


def test_complex_pattern_round_trip(tmp_path: Path) -> None:
    """
    Integration test: Complex pattern generation → Save → Load → Verify.

    Workflow:
    1. Generate complex pattern using multiple methods
    2. Save to file
    3. Load and analyze
    4. Verify statistical properties match expectations

    Design decision: Test that complex generation workflows are reliable.
    """
    image_path = tmp_path / "complex_pattern.png"

    # Step 1: Generate using mix of methods
    gen = ImageGenerator(128, 128)

    # Fill background
    gen.fill(128, 128, 128)

    # Add random noise to top half
    rng = np.random.RandomState(42)
    for y in range(64):
        for x in range(128):
            gen.set_pixel(
                x, y, rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)
            )

    # Add gradient to bottom half
    for y in range(64, 128):
        for x in range(128):
            gen.set_pixel(x, y, x * 2, (y - 64) * 4, 128)

    # Step 2: Save
    gen.save(str(image_path))

    # Step 3: Load and analyze
    metrics = ImageMetrics(str(image_path))

    # Step 4: Verify expected properties
    # Top half is random, bottom half is gradient
    # Overall should have moderate-high entropy
    entropy = metrics.entropy(ImageMetrics.RGB.ALL)
    assert 6.0 < entropy < 8.0, (
        f"Mixed pattern should have moderate entropy, got {entropy:.4f}"
    )

    # Correlation should be moderate (gradient contributes correlation)
    corr_v = metrics.correlation(ImageMetrics.RGB.ALL, "vertical")
    assert abs(corr_v) > 0.2, (
        f"Mixed pattern should show some vertical correlation, got {corr_v:.4f}"
    )

    # Should have good variety of unique values
    unique = len(np.unique(metrics.image))
    assert unique > 250, (
        f"Complex pattern should have variety, got {unique} unique values"
    )


# ============================================================================
# Performance/Stress Tests (Optional - can be marked as slow)
# ============================================================================


@pytest.mark.slow
def test_large_image_workflow(tmp_path: Path) -> None:
    """
    Integration test: Large image (HD resolution) full workflow.

    Workflow:
    1. Generate 1920x1080 image
    2. Run full analysis
    3. Verify completes without error

    Design decision: Verify scalability to real-world image sizes.
    Note: Marked as 'slow' - run with: pytest -m slow
    """
    image_path = tmp_path / "large_image.png"

    # Step 1: Generate HD image
    gen = ImageGenerator(1920, 1080)
    rng = np.random.RandomState(111)

    # Fill using byte-level operations (faster)
    for i in range(gen.byte_count):
        gen.set_byte(i, rng.randint(0, 256))

    gen.save(str(image_path))

    # Step 2: Analyze (should handle large data)
    metrics = ImageMetrics(str(image_path))
    analysis = metrics.analyze_all(ImageMetrics.RGB.ALL)

    # Step 3: Verify reasonable results
    assert analysis["total_bytes"] == 1920 * 1080 * 3, (
        "Byte count should match image size"
    )

    assert analysis["entropy"] > 7.8, "Large random image should have high entropy"

    # Monte Carlo should have good accuracy with this many points
    assert analysis["monte_carlo_pi"]["error_percentage"] < 5.0, (
        f"Large sample should estimate π well, got {analysis['monte_carlo_pi']['error_percentage']:.2f}% error"
    )
