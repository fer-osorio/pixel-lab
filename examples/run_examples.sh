#!/usr/bin/env bash

################################################################################
# run_examples.sh
#
# Automated execution script for Pixel Lab examples
#
# Design decisions:
# 1. Use bash over sh for better error handling and array support
# 2. Strict mode (set -euo pipefail) for robustness
# 3. Color output for better UX
# 4. Validate inputs before execution
# 5. Report summary at the end
################################################################################

# Strict mode: exit on error, undefined variables, and pipe failures
set -euo pipefail

# Script configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly EXAMPLES_DIR="${SCRIPT_DIR}"

# Color codes for output (design: improve readability)
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Counters for summary report
SUCCESS_COUNT=0
FAILURE_COUNT=0
SKIPPED_COUNT=0

################################################################################
# Helper Functions
################################################################################

# Design decision: Separate display logic for maintainability
print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Design decision: Centralized help text for single source of truth
show_help() {
    cat << EOF
Pixel Lab Examples Runner

USAGE:
    ./run_examples.sh [OPTIONS] [FILE]

DESCRIPTION:
    Execute Python example scripts with automatic validation and reporting.
    All examples must have a .py extension and be located in the examples/ directory.

OPTIONS:
    -h, --help              Show this help message and exit
    -a, --all               Run all Python scripts in examples/
    -p, --pixel             Run all scripts with 'pixel' in the filename
    -b, --byte              Run all scripts with 'byte' in the filename
    -d, --direct            Run all scripts with 'direct' in the filename
    -r, --recursive         Run all scripts with 'recursive' in the filename
    -v, --verbose           Show detailed output from each script

ARGUMENTS:
    FILE                    Specific Python script to run (e.g., example.py)

EXAMPLES:
    # Run a specific example
    ./run_examples.sh 01_basic_generation.py

    # Run all examples
    ./run_examples.sh --all

    # Run all pixel-related examples
    ./run_examples.sh --pixel

    # Run all recursive examples with verbose output
    ./run_examples.sh --recursive --verbose

NOTES:
    - Scripts are executed from the examples/ directory
    - Exit codes: 0 = success, 1 = validation error, 2 = execution error
    - A summary report is shown after execution

EOF
}

################################################################################
# Validation Functions
################################################################################

# Design decision: Early validation prevents cryptic errors during execution
validate_python_script() {
    local file="$1"

    # Check if file exists
    if [[ ! -f "${file}" ]]; then
        print_error "File not found: ${file}"
        return 1
    fi

    # Check .py extension (case-insensitive for robustness)
    if [[ ! "${file}" =~ \.py$ ]]; then
        print_error "Not a Python script (missing .py extension): ${file}"
        return 1
    fi

    # Check if file is readable
    if [[ ! -r "${file}" ]]; then
        print_error "File not readable: ${file}"
        return 1
    fi

    # Optional: Check if file has valid Python syntax
    # Design decision: Use python -m py_compile for syntax check
    if ! python3 -m py_compile "${file}" 2>/dev/null; then
        print_warning "File has syntax errors: ${file}"
        # Don't return error - let user decide if they want to run it anyway
    fi

    return 0
}

# Design decision: Verify Python environment before running scripts
check_python_environment() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed or not in PATH"
        exit 1
    fi

    # Optional: Check if pixel_lab is importable
    if ! python3 -c "import pixel_lab" 2>/dev/null; then
        print_warning "pixel_lab package not found. Have you installed it?"
        print_info "Run: pip install -e . from the project root"
        # Don't exit - examples might still work with local imports
    fi
}

################################################################################
# Execution Functions
################################################################################

# Design decision: Centralized execution logic with consistent error handling
run_script() {
    local script="$1"
    local verbose="${2:-false}"

    print_info "Running: ${script}"

    # Design decision: Run from examples directory to match expected working directory
    if [[ "${verbose}" == "true" ]]; then
        # Show full output in verbose mode
        if python3 "${script}"; then
            print_success "Completed: ${script}"
            ((SUCCESS_COUNT++))
            return 0
        else
            print_error "Failed: ${script}"
            ((FAILURE_COUNT++))
            return 1
        fi
    else
        # Suppress output in normal mode, only show summary
        if python3 "${script}" > /dev/null 2>&1; then
            print_success "Completed: ${script}"
            ((SUCCESS_COUNT++))
            return 0
        else
            print_error "Failed: ${script}"
            ((FAILURE_COUNT++))
            return 1
        fi
    fi
}

# Design decision: Use arrays for efficient file collection and iteration
find_and_run_scripts() {
    local pattern="$1"
    local verbose="${2:-false}"
    local scripts=()

    # Find all matching Python files
    # Design decision: Use -name for filename matching, more portable than regex
    while IFS= read -r -d '' file; do
        scripts+=("${file}")
    done < <(find "${EXAMPLES_DIR}" -maxdepth 1 -type f -name "*${pattern}*.py" -print0 | sort -z)

    # Check if any scripts found
    if [[ ${#scripts[@]} -eq 0 ]]; then
        print_warning "No scripts found matching pattern: *${pattern}*.py"
        return 0
    fi

    print_header "Found ${#scripts[@]} script(s) matching pattern: ${pattern}"

    # Execute each script
    for script in "${scripts[@]}"; do
        # Design decision: Continue on failure to run all scripts
        run_script "${script}" "${verbose}" || true
    done
}

# Design decision: Separate function for running all scripts (clearer logic)
run_all_scripts() {
    local verbose="${1:-false}"
    local scripts=()

    # Find all Python files
    while IFS= read -r -d '' file; do
        scripts+=("${file}")
    done < <(find "${EXAMPLES_DIR}" -maxdepth 1 -type f -name "*.py" -print0 | sort -z)

    if [[ ${#scripts[@]} -eq 0 ]]; then
        print_warning "No Python scripts found in ${EXAMPLES_DIR}"
        return 0
    fi

    print_header "Running all ${#scripts[@]} example script(s)"

    for script in "${scripts[@]}"; do
        run_script "${script}" "${verbose}" || true
    done
}

################################################################################
# Summary Report
################################################################################

# Design decision: Always show summary for user feedback
print_summary() {
    echo ""
    print_header "Execution Summary"

    local total=$((SUCCESS_COUNT + FAILURE_COUNT + SKIPPED_COUNT))

    echo -e "Total scripts: ${total}"
    echo -e "${GREEN}Successful:${NC}    ${SUCCESS_COUNT}"
    echo -e "${RED}Failed:${NC}        ${FAILURE_COUNT}"

    if [[ ${SKIPPED_COUNT} -gt 0 ]]; then
        echo -e "${YELLOW}Skipped:${NC}       ${SKIPPED_COUNT}"
    fi

    echo ""

    # Design decision: Exit with error code if any failures occurred
    if [[ ${FAILURE_COUNT} -gt 0 ]]; then
        print_error "Some scripts failed. Check output above for details."
        return 2
    else
        print_success "All scripts completed successfully!"
        return 0
    fi
}

################################################################################
# Main Logic
################################################################################

main() {
    # Default values
    local verbose=false
    local mode=""
    local target_file=""

    # Design decision: Parse arguments first, validate later
    if [[ $# -eq 0 ]]; then
        show_help
        exit 0
    fi

    # Parse command-line arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                show_help
                exit 0
                ;;
            -a|--all)
                mode="all"
                shift
                ;;
            -p|--pixel)
                mode="pixel"
                shift
                ;;
            -b|--byte)
                mode="byte"
                shift
                ;;
            -d|--direct)
                mode="direct"
                shift
                ;;
            -r|--recursive)
                mode="recursive"
                shift
                ;;
            -v|--verbose)
                verbose=true
                shift
                ;;
            -*)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
            *)
                # Design decision: Assume any non-option is a filename
                if [[ -n "${mode}" ]]; then
                    print_error "Cannot specify both a pattern and a specific file"
                    exit 1
                fi
                target_file="$1"
                mode="single"
                shift
                ;;
        esac
    done

    # Validate Python environment
    check_python_environment

    # Change to examples directory
    # Design decision: Run from examples/ so relative paths in scripts work
    cd "${EXAMPLES_DIR}"

    print_header "Pixel Lab Examples Runner"

    # Execute based on mode
    case "${mode}" in
        single)
            # Validate and run single file
            if validate_python_script "${target_file}"; then
                run_script "${target_file}" "${verbose}"
            else
                exit 1
            fi
            ;;
        all)
            run_all_scripts "${verbose}"
            ;;
        pixel|byte|direct|recursive)
            find_and_run_scripts "${mode}" "${verbose}"
            ;;
        *)
            print_error "No mode specified"
            show_help
            exit 1
            ;;
    esac

    # Show summary and exit with appropriate code
    print_summary
    exit $?
}

# Design decision: Use trap to ensure summary is shown even on early exit
trap 'print_summary' EXIT

# Entry point
main "$@"
