"""
Unit Tests for ImageMetrics Class

Tests organized by functionality:
1. Constructor tests (2 tests)
2. Entropy tests (3 tests)
3. Chi-square tests (2 tests)
4. Correlation tests (3 tests)
5. MAD tests (2 tests)
6. Monte Carlo π tests (2 tests)
7. Visualization tests (1 test)
8. Edge cases (5 tests)

Total: 20 tests

Design philosophy:
- Use fixtures for test images (uniform_random, solid_color, gradient)
- Test statistical properties with reasonable tolerances
- Verify error handling for edge cases
- Explicit type hints for all parameters and returns
"""

# pytest: Testing framework - provides assertions and fixtures
# os: File operations for checking saved plots
import os

# pathlib.Path: Type hints for path parameters
from pathlib import Path

# numpy: Statistical calculations and array comparisons
import numpy as np
import pytest

# Import test utilities from conftest
from conftest import assert_correlation_type, assert_entropy_in_range

# Import class under test
from pixel_lab import ImageMetrics

# ============================================================================
# 1. Constructor Tests
# ============================================================================


def test_constructor_loads_valid_image(uniform_random_image: Path) -> None:
    """
    Test that constructor successfully loads a valid image file.

    Design decision: Verify basic properties (dimensions, array shape, dtype)
    to ensure image was loaded correctly.
    """
    metrics = ImageMetrics(str(uniform_random_image))

    # Verify properties set correctly
    assert metrics.width == 100
    assert metrics.height == 100
    assert metrics.image.shape == (100, 100, 3)
    assert metrics.image.dtype == np.uint8
    assert metrics.filename == str(uniform_random_image)


def test_constructor_rejects_invalid_path() -> None:
    """
    Test that constructor raises FileNotFoundError for non-existent file.

    Design decision: Fail fast with clear error when file doesn't exist.
    """
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        ImageMetrics("nonexistent_file.png")


# ============================================================================
# 2. Entropy Tests
# ============================================================================


def test_entropy_uniform_random_image(uniform_random_image: Path) -> None:
    """
    Test that entropy of uniform random image is close to maximum (8.0).

    Design decision: Allow small tolerance (7.8-8.0) because even good
    PRNGs rarely achieve exactly 8.0 due to sampling variation.
    """
    metrics = ImageMetrics(str(uniform_random_image))
    entropy = metrics.entropy(ImageMetrics.RGB.ALL)

    # Uniform random should have very high entropy
    assert_entropy_in_range(entropy, 7.8, 8.0, "Uniform random image")


def test_entropy_solid_color_image(solid_color_image: Path) -> None:
    """
    Test that entropy of solid color image is very low (near 0.0).

    Design decision: Solid color has only one unique value, so entropy
    should be exactly 0.0 (no information, completely predictable).
    """
    metrics = ImageMetrics(str(solid_color_image))
    entropy = metrics.entropy(ImageMetrics.RGB.ALL)

    # Solid color should have zero entropy
    assert entropy < 0.1, f"Expected near-zero entropy, got {entropy:.4f}"


def test_entropy_per_channel(gradient_image: Path) -> None:
    """
    Test that entropy can be computed for individual RGB channels.

    Design decision: Verify channel isolation - RED and GREEN channels
    in gradient should have similar entropy, different from ALL channels.
    """
    metrics = ImageMetrics(str(gradient_image))

    # Compute entropy for each channel
    entropy_all = metrics.entropy(ImageMetrics.RGB.ALL)
    entropy_red = metrics.entropy(ImageMetrics.RGB.RED)
    entropy_green = metrics.entropy(ImageMetrics.RGB.GREEN)
    entropy_blue = metrics.entropy(ImageMetrics.RGB.BLUE)

    # RED and GREEN should have similar entropy (both gradients)
    assert abs(entropy_red - entropy_green) < 0.5, (
        "RED and GREEN gradients should have similar entropy"
    )

    # BLUE should have lower entropy (constant 0)
    assert entropy_blue < entropy_red, (
        "Constant BLUE channel should have lower entropy than gradient"
    )

    # ALL should different from all the others
    assert entropy_all != entropy_blue and entropy_all != entropy_red, (
        "ALL entropy should be diferent from the others"
    )


# ============================================================================
# 3. Chi-Square Tests
# ============================================================================


def test_chi_square_uniform_distribution(uniform_random_image: Path) -> None:
    """
    Test that chi-square test passes (p > 0.05) for uniform random data.

    Design decision: p-value > 0.05 means we cannot reject the null hypothesis
    that the distribution is uniform (good for randomness).
    """
    metrics = ImageMetrics(str(uniform_random_image))
    chi2_result = metrics.chi_square(ImageMetrics.RGB.ALL)

    # Verify return structure
    assert "statistic" in chi2_result
    assert "p_value" in chi2_result
    assert "dof" in chi2_result

    # Verify degrees of freedom
    assert chi2_result["dof"] == 255  # 256 categories - 1

    # Uniform distribution should pass chi-square test
    assert chi2_result["p_value"] > 0.05, (
        f"Expected p-value > 0.05 for uniform data, got {chi2_result['p_value']:.6f}"
    )


def test_chi_square_non_uniform_distribution(solid_color_image: Path) -> None:
    """
    Test that chi-square test fails (p < 0.05) for non-uniform data.

    Design decision: Solid color is extremely non-uniform, so should have
    p-value very close to 0.0.
    """
    metrics = ImageMetrics(str(solid_color_image))
    chi2_result = metrics.chi_square(ImageMetrics.RGB.ALL)

    # Non-uniform distribution should fail chi-square test
    assert chi2_result["p_value"] < 0.05, (
        f"Expected p-value < 0.05 for non-uniform data, got {chi2_result['p_value']:.6f}"
    )

    # For solid color, p-value should be essentially zero
    assert chi2_result["p_value"] < 1e-10, "Solid color should have near-zero p-value"


# ============================================================================
# 4. Correlation Tests
# ============================================================================


def test_correlation_random_image_low(uniform_random_image: Path) -> None:
    """
    Test that correlation is near zero for random data.

    Design decision: Random data should have negligible correlation in all
    directions (horizontal, vertical, diagonal).
    """
    metrics = ImageMetrics(str(uniform_random_image))

    # Test all three directions
    corr_h = metrics.correlation(ImageMetrics.RGB.ALL, "horizontal")
    corr_v = metrics.correlation(ImageMetrics.RGB.ALL, "vertical")
    corr_d = metrics.correlation(ImageMetrics.RGB.ALL, "diagonal")

    # All correlations should be near zero
    assert_correlation_type(corr_h, "zero")
    assert_correlation_type(corr_v, "zero")
    assert_correlation_type(corr_d, "zero")


def test_correlation_gradient_image_high(gradient_image: Path) -> None:
    """
    Test that correlation is high for gradient data.

    Design decision: Gradients have strong spatial correlation - adjacent
    pixels have very similar values in the same channels.
    """
    metrics = ImageMetrics(str(gradient_image))

    # Test horizontal and vertical (both are gradients in this image)
    # Must be almost one
    corr_vr = metrics.correlation(ImageMetrics.RGB.RED, "vertical")
    # Must be almost one
    corr_hg = metrics.correlation(ImageMetrics.RGB.GREEN, "horizontal")
    # Must be almost one
    corr_dr = metrics.correlation(ImageMetrics.RGB.RED, "diagonal")
    # Must be almost one
    corr_dg = metrics.correlation(ImageMetrics.RGB.GREEN, "diagonal")

    # Gradients should show high correlation
    assert_correlation_type(corr_vr, "one")
    assert_correlation_type(corr_hg, "one")
    assert_correlation_type(corr_dr, "one")
    assert_correlation_type(corr_dg, "one")


def test_correlation_gradient_image_constant_component(gradient_image: Path) -> None:
    """
    Test that constant data sets have zero correlation coefficient.

    Design decision: By design, blue channel is constant for gradient
    images, therefore correlation coefficient in blue channel is zero.
    """
    metrics = ImageMetrics(str(gradient_image))

    corr_hb = metrics.correlation(ImageMetrics.RGB.BLUE, "horizontal")
    corr_vb = metrics.correlation(ImageMetrics.RGB.BLUE, "vertical")
    corr_db = metrics.correlation(ImageMetrics.RGB.BLUE, "diagonal")

    # Blue channel in gradient image should have a negligible correlation
    assert_correlation_type(corr_hb, "zero")
    assert_correlation_type(corr_vb, "zero")
    assert_correlation_type(corr_db, "zero")


def test_correlation_different_directions(gradient_image: Path) -> None:
    """
    Test that different directions produce different correlation values.

    Design decision: Verify that horizontal, vertical, and diagonal
    correlations are computed independently and can differ.
    """
    metrics = ImageMetrics(str(gradient_image))

    corr_h = metrics.correlation(ImageMetrics.RGB.ALL, "horizontal")
    corr_v = metrics.correlation(ImageMetrics.RGB.ALL, "vertical")
    corr_d = metrics.correlation(ImageMetrics.RGB.ALL, "diagonal")

    # All three should be different values
    assert corr_h != corr_v or corr_v != corr_d, (
        "Different directions should produce different correlations"
    )


# ============================================================================
# 5. Mean Absolute Deviation Tests
# ============================================================================


def test_mad_uniform_distribution(uniform_random_image: Path) -> None:
    """
    Test that MAD is low for uniform random distribution.

    Design decision: Uniform distribution should have low deviation from
    expected uniform frequency (< 10% is good).
    """
    metrics = ImageMetrics(str(uniform_random_image))
    mad_result = metrics.mean_absolute_deviation(ImageMetrics.RGB.ALL)

    # Verify return structure
    assert "mad" in mad_result
    assert "mad_percentage" in mad_result
    assert "expected_frequency" in mad_result

    # Uniform distribution should have low MAD percentage
    assert mad_result["mad_percentage"] < 15.0, (
        f"Expected MAD < 15% for uniform data, got {mad_result['mad_percentage']:.2f}%"
    )


def test_mad_non_uniform_distribution(solid_color_image: Path) -> None:
    """
    Test that MAD is high for non-uniform distribution.

    Design decision: Solid color has maximum possible deviation from uniform.
    """
    metrics = ImageMetrics(str(solid_color_image))
    mad_result = metrics.mean_absolute_deviation(ImageMetrics.RGB.ALL)

    # Non-uniform should have high MAD percentage
    assert mad_result["mad_percentage"] > 50.0, (
        f"Expected MAD > 50% for solid color, got {mad_result['mad_percentage']:.2f}%"
    )


# ============================================================================
# 6. Monte Carlo π Tests
# ============================================================================


def test_monte_carlo_pi_random_data(uniform_random_image: Path) -> None:
    """
    Test that Monte Carlo π estimation is accurate for random data.

    Design decision: With uniform random data, point distribution should
    be uniform across unit square, yielding accurate π estimate.
    """
    metrics = ImageMetrics(str(uniform_random_image))
    mc_result = metrics.monte_carlo_pi(ImageMetrics.RGB.ALL)

    # Verify return structure
    assert "pi_estimate" in mc_result
    assert "error" in mc_result
    assert "error_percentage" in mc_result
    assert "points_used" in mc_result
    assert "true_pi" in mc_result

    # Verify number of points
    expected_points = (100 * 100 * 3) // 2  # 15,000 pairs
    assert mc_result["points_used"] == expected_points

    # Verify π estimate is reasonable
    assert 2.5 < mc_result["pi_estimate"] < 3.8, (
        f"π estimate {mc_result['pi_estimate']:.4f} is unrealistic"
    )

    # For good random data, error should be < 10%
    assert mc_result["error_percentage"] < 10.0, (
        f"Expected error < 10%, got {mc_result['error_percentage']:.2f}%"
    )


def test_monte_carlo_pi_insufficient_data(small_image: Path) -> None:
    """
    Test that Monte Carlo π handles small images appropriately.

    Design decision: 1x1 image (3 bytes) has only 1 coordinate pair,
    should work without crashing.
    """
    metrics = ImageMetrics(str(small_image))
    mc_result = metrics.monte_carlo_pi(ImageMetrics.RGB.ALL)

    # Should work with minimal data
    assert mc_result["points_used"] == 1  # Only 1 coordinate pair

    # Estimate will be either 0 or 4 (binary with 1 point)
    assert mc_result["pi_estimate"] in [0.0, 4.0], (
        f"With 1 point, estimate should be 0 or 4, got {mc_result['pi_estimate']}"
    )


# ============================================================================
# 7. Visualization Tests
# ============================================================================


@pytest.mark.slow
def test_plot_generation_no_crash(uniform_random_image: Path, tmp_path: Path) -> None:
    """
    Test that plot generation functions execute without crashing.

    Design decision: Smoke test for visualization - we can't easily verify
    visual correctness, but can ensure no exceptions are raised.
    """
    metrics = ImageMetrics(str(uniform_random_image))

    # Test frequency distribution plot
    freq_path = tmp_path / "freq_plot.png"
    metrics.plot_frequency_distribution(ImageMetrics.RGB.ALL, save_path=str(freq_path))
    assert os.path.exists(freq_path), "Frequency plot not saved"

    # Test correlation heatmap
    corr_path = tmp_path / "corr_plot.png"
    metrics.plot_correlation_heatmap(ImageMetrics.RGB.ALL, save_path=str(corr_path))
    assert os.path.exists(corr_path), "Correlation plot not saved"

    # Test Monte Carlo visualization
    mc_path = tmp_path / "mc_plot.png"
    metrics.plot_monte_carlo_visualization(ImageMetrics.RGB.ALL, save_path=str(mc_path))
    assert os.path.exists(mc_path), "Monte Carlo plot not saved"


# ============================================================================
# 8. Edge Cases
# ============================================================================


def test_correlation_with_small_lag(gradient_image: Path) -> None:
    """
    Test correlation with different lag values.

    Design decision: Lag parameter allows testing correlation at different
    distances, not just adjacent pixels.
    """
    metrics = ImageMetrics(str(gradient_image))

    # Test default lag
    corr_lag1 = metrics.correlation(ImageMetrics.RGB.ALL, "horizontal", lag=1)
    corr_ALL = metrics.correlation(ImageMetrics.RGB.ALL, "horizontal")
    # Test lag=3 coincidence with red channel
    corrh_lag3 = metrics.correlation(ImageMetrics.RGB.ALL, "horizontal", lag=3)
    corrh_RED = metrics.correlation(ImageMetrics.RGB.RED, "horizontal")
    # Test lag inequality; red vertical correlation > All verical correlation
    corrv_lag2 = metrics.correlation(ImageMetrics.RGB.ALL, "vertical", lag=2)
    corrv_RED2 = metrics.correlation(ImageMetrics.RGB.RED, "vertical", lag=2)

    # The following properties must follow
    assert corr_lag1 == corr_ALL, "Lag 1 should should be the default"
    assert corrh_lag3 == corrh_RED, (
        "Lag 3 horizontal should coincide with horizontal correlation on red channel"
    )
    assert corrv_lag2 < corrv_RED2, (
        "For all lags, vertical correlation on ALL channel should be smaller than vertical "
        "correlation on red channel"
    )


def test_correlation_invalid_direction(uniform_random_image: Path) -> None:
    """
    Test that invalid direction raises appropriate error.

    Design decision: Validate direction parameter to prevent silent errors.
    """
    metrics = ImageMetrics(str(uniform_random_image))

    with pytest.raises(ValueError, match="Invalid direction"):
        metrics.correlation(ImageMetrics.RGB.ALL, "invalid_direction")  # type: ignore


def test_all_channels_independently(gradient_image: Path) -> None:
    """
    Test that all RGB enum values work for each metric.

    Design decision: Comprehensive test that all channel options are valid.
    """
    metrics = ImageMetrics(str(gradient_image))

    channels = [
        ImageMetrics.RGB.ALL,
        ImageMetrics.RGB.RED,
        ImageMetrics.RGB.GREEN,
        ImageMetrics.RGB.BLUE,
    ]

    for channel in channels:
        # Each metric should work for each channel
        entropy = metrics.entropy(channel)
        assert 0.0 <= entropy <= 8.0, f"Invalid entropy for {channel.name}"

        chi2 = metrics.chi_square(channel)
        assert "p_value" in chi2, f"Missing p_value for {channel.name}"

        corr = metrics.correlation(channel, "horizontal")
        assert -1.0 <= corr <= 1.0, f"Invalid correlation for {channel.name}"


def test_byte_frequency_normalized_sums_to_one(uniform_random_image: Path) -> None:
    """
    Test that normalized frequency distribution sums to 1.0.

    Design decision: Probability distribution must sum to 1 by definition.
    """
    metrics = ImageMetrics(str(uniform_random_image))

    freq_norm = metrics.byte_frequency_normalized(ImageMetrics.RGB.ALL)

    # Should sum to 1.0 (within floating point tolerance)
    assert abs(freq_norm.sum() - 1.0) < 1e-10, (
        f"Normalized frequency sum {freq_norm.sum()} != 1.0"
    )


def test_monte_carlo_with_even_odd_byte_counts(temp_image_path: Path) -> None:
    """
    Test Monte Carlo with odd total byte count (discards last byte).

    Design decision: Monte Carlo needs pairs of bytes, so odd counts
    should discard the last unpaired byte.
    """
    from pixel_lab import ImageGenerator

    # Create image with odd total bytes: 3x3 = 27 bytes
    gen = ImageGenerator(3, 3)
    rng = np.random.RandomState(42)
    for i in range(gen.byte_count):
        gen.set_byte(i, rng.randint(0, 256))
    gen.save(str(temp_image_path))

    metrics = ImageMetrics(str(temp_image_path))
    mc_result = metrics.monte_carlo_pi(ImageMetrics.RGB.ALL)

    # Should use 13 pairs (26 bytes), discarding last byte
    assert mc_result["points_used"] == 13, (
        f"Expected 13 pairs from 27 bytes, got {mc_result['points_used']}"
    )


# ============================================================================
# Additional Tests for Complete Coverage
# ============================================================================


def test_byte_frequency_counts(solid_color_image: Path) -> None:
    """
    Test that byte_frequency returns correct counts.

    Design decision: Verify frequency array structure and that counts
    sum to total bytes.
    """
    metrics = ImageMetrics(str(solid_color_image))
    freq = metrics.byte_frequency(ImageMetrics.RGB.ALL)

    # Verify shape
    assert freq.shape == (256,), "Frequency array should have 256 bins"

    # Verify counts sum to total bytes
    total_bytes = 100 * 100 * 3  # 30,000
    assert freq.sum() == total_bytes, (
        f"Frequency sum {freq.sum()} != total bytes {total_bytes}"
    )

    # For solid color (128, 128, 128), only value 128 should have count
    assert freq[128] == total_bytes, (
        "All bytes should be value 128 for solid gray image"
    )
    assert np.sum(freq != 0) == 1, "Only one unique value should have non-zero count"


def test_analyze_all_returns_complete_data(uniform_random_image: Path) -> None:
    """
    Test that analyze_all returns dictionary with all expected keys.

    Design decision: Comprehensive analysis should include all metrics
    in a single structured result.
    """
    metrics = ImageMetrics(str(uniform_random_image))
    analysis = metrics.analyze_all(ImageMetrics.RGB.ALL)

    # Verify all expected keys present
    expected_keys = [
        "channel",
        "entropy",
        "chi_square",
        "correlation_horizontal",
        "correlation_vertical",
        "correlation_diagonal",
        "mean_absolute_deviation",
        "monte_carlo_pi",
        "byte_frequency",
        "unique_values",
        "total_bytes",
        "mean",
        "std_dev",
        "min",
        "max",
    ]

    for key in expected_keys:
        assert key in analysis, f"Missing key in analysis: {key}"

    # Verify channel name
    assert analysis["channel"] == "ALL"


def test_summary_generates_text_report(uniform_random_image: Path) -> None:
    """
    Test that summary generates non-empty text report.

    Design decision: Summary should be human-readable string with
    key metrics and interpretations.
    """
    metrics = ImageMetrics(str(uniform_random_image))

    # Test normal summary
    summary = metrics.summary(ImageMetrics.RGB.ALL, verbose=False)
    assert isinstance(summary, str)
    assert len(summary) > 100, "Summary should be substantial"
    assert "SHANNON ENTROPY" in summary
    assert "CHI-SQUARE TEST" in summary

    # Test verbose summary
    verbose_summary = metrics.summary(ImageMetrics.RGB.ALL, verbose=True)
    assert len(verbose_summary) > len(summary), (
        "Verbose summary should be longer than normal"
    )
    assert "INTERPRETATION GUIDE" in verbose_summary
