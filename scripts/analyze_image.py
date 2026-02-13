# ============================================================================
# Command-Line Interface
# ============================================================================

# argparse: Command-line argument parsing for CLI interface
import argparse

# sys: System-specific parameters (stderr, exit codes) for CLI error handling
import sys

from pixel_lab import ImageMetrics


def main():
    """
    CLI entry point for analyzing images from the command line.

    Design decision: Provide a simple CLI for quick analysis without
    writing Python code. Useful for batch processing or integration
    with other tools.
    """
    parser = argparse.ArgumentParser(
        description="Analyze statistical and cryptanalytic properties of images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.png
  %(prog)s image.png --channel red
  %(prog)s image.png --channel all --verbose
  %(prog)s image.png --metrics entropy chi_square correlation
  %(prog)s image.png --plot
  %(prog)s image.png --save-plots ./output_plots

Channel options: all, red, green, blue
Metric options: entropy, chi_square, correlation, mad, monte_carlo, all
        """,
    )

    parser.add_argument("image", help="Path to image file")

    parser.add_argument(
        "--channel",
        "-c",
        choices=["all", "red", "green", "blue"],
        default="all",
        help="Color channel to analyze (default: all)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed interpretation guide",
    )

    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        choices=["entropy", "chi_square", "correlation", "mad", "monte_carlo", "all"],
        default=["all"],
        help="Specific metrics to compute (default: all)",
    )

    parser.add_argument(
        "--plot", "-p", action="store_true", help="Generate visualization plots"
    )

    parser.add_argument(
        "--save-plots",
        "-s",
        type=str,
        metavar="DIR",
        help="Save plots to specified directory instead of displaying",
    )

    args = parser.parse_args()

    # Map string argument to enum
    channel_map = {
        "all": ImageMetrics.RGB.ALL,
        "red": ImageMetrics.RGB.RED,
        "green": ImageMetrics.RGB.GREEN,
        "blue": ImageMetrics.RGB.BLUE,
    }
    channel = channel_map[args.channel]

    try:
        # Load image
        metrics = ImageMetrics(args.image)

        # Determine which metrics to show
        show_all = "all" in args.metrics

        if show_all:
            # Show complete summary
            print(metrics.summary(channel, verbose=args.verbose))

            # Generate plots if requested
            if args.plot or args.save_plots:
                metrics.plot_all(channel, save_dir=args.save_plots)
        else:
            # Show only requested metrics
            print(f"\n{'=' * 70}")
            print(f"Image: {args.image}")
            print(f"Channel: {channel.name}")
            print("=" * 70)

            if "entropy" in args.metrics:
                entropy = metrics.entropy(channel)
                print(f"\nEntropy: {entropy:.6f} bits")

            if "chi_square" in args.metrics:
                chi2 = metrics.chi_square(channel)
                print(f"\nChi-Square:")
                print(f"  Statistic: {chi2['statistic']:.2f}")
                print(f"  P-Value: {chi2['p_value']:.6f}")
                print(f"  Result: {'PASS' if chi2['p_value'] > 0.05 else 'FAIL'}")

            if "mad" in args.metrics:
                mad = metrics.mean_absolute_deviation(channel)
                print(f"\nMean Absolute Deviation:")
                print(f"  MAD: {mad['mad']:.2f}")
                print(f"  MAD %: {mad['mad_percentage']:.2f}%")

            if "monte_carlo" in args.metrics:
                mc = metrics.monte_carlo_pi(channel)
                print(f"\nMonte Carlo π Estimation:")
                print(f"  Estimated π: {mc['pi_estimate']:.6f}")
                print(f"  Error: {mc['error_percentage']:.2f}%")

            if "correlation" in args.metrics:
                corr_h = metrics.correlation(channel, "horizontal")
                corr_v = metrics.correlation(channel, "vertical")
                corr_d = metrics.correlation(channel, "diagonal")
                print(f"\nCorrelation:")
                print(f"  Horizontal: {corr_h:+.6f}")
                print(f"  Vertical:   {corr_v:+.6f}")
                print(f"  Diagonal:   {corr_d:+.6f}")

            print("=" * 70 + "\n")

            # Generate plots if requested
            if args.plot or args.save_plots:
                print("\nGenerating visualizations...")
                metrics.plot_all(channel, save_dir=args.save_plots)

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
COMMAND-LINE USAGE:
-------------------
python analyze_image.py <image_file> [options]

Options:
  --channel, -c   : Specify channel (all, red, green, blue)
  --verbose, -v   : Show detailed interpretation guide
  --metrics, -m   : Specific metrics (entropy, chi_square, correlation, mad, monte_carlo, all)
  --plot, -p      : Generate and display visualization plots
  --save-plots, -s: Save plots to directory instead of displaying

Examples:
  python analyze_image.py image.png
  python analyze_image.py image.png --channel red --verbose
  python analyze_image.py image.png -m entropy mad monte_carlo
  python analyze_image.py image.png --plot
  python analyze_image.py image.png --save-plots ./analysis_output
        """)
