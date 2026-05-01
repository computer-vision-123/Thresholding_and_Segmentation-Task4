#!/usr/bin/env bash
# =============================================================================
# Usage:
#   ./build.sh            # full build (venv + deps + C++ extension)
#   ./build.sh --deps     # install Python dependencies only
#   ./build.sh --cpp      # compile the C++ pybind11 extension only
#   ./build.sh --run      # build then launch the application
#   ./build.sh --clean    # remove build artefacts and the .so extension
# =============================================================================

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'   # No Colour

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERR]${NC}   $*" >&2; }

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

# ── Parse arguments ───────────────────────────────────────────────────────────
MODE="all"
case "${1:-}" in
    --deps)  MODE="deps"  ;;
    --cpp)   MODE="cpp"   ;;
    --run)   MODE="run"   ;;
    --clean) MODE="clean" ;;
    "")      MODE="all"   ;;
    *)
        error "Unknown option: ${1}"
        echo "Usage: $0 [--deps | --cpp | --run | --clean]"
        exit 1
        ;;
esac

# =============================================================================
# Helper: ensure virtual environment exists
# =============================================================================
ensure_venv() {
    if [[ ! -d "${VENV_DIR}" ]]; then
        info "Creating Python virtual environment at .venv …"
        python3 -m venv "${VENV_DIR}"
        success "Virtual environment created."
    else
        info "Virtual environment already exists – skipping creation."
    fi
}

# =============================================================================
# Step 1 – Install Python dependencies
# =============================================================================
install_deps() {
    ensure_venv
    info "Upgrading pip …"
    "${PIP}" install --quiet --upgrade pip

    info "Installing Python dependencies from requirements.txt …"
    "${PIP}" install --quiet -r "${SCRIPT_DIR}/requirements.txt"
    success "Python dependencies installed."
}

# =============================================================================
# Step 2 – Build the C++ pybind11 extension (cv_backend)
# =============================================================================
build_cpp() {
    ensure_venv
    info "Building C++ extension (cv_backend) via scikit-build-core …"

    # pip install in editable/no-build-isolation mode so that the compiled
    # .so lands in the project root, making `import cv_backend` work directly.
    "${PIP}" install --quiet \
        --no-build-isolation \
        --config-settings=cmake.build-type=Release \
        -e "${SCRIPT_DIR}"

    # Verify the extension was produced
    SO_FILE=$(find "${SCRIPT_DIR}" -maxdepth 4 -name "cv_backend*.so" | head -n1)
    if [[ -n "${SO_FILE}" ]]; then
        # If it's not in the root, copy it there for easier import
        if [[ "$(dirname "${SO_FILE}")" != "${SCRIPT_DIR}" ]]; then
            cp "${SO_FILE}" "${SCRIPT_DIR}/"
            success "Extension built and copied to root: $(basename "${SO_FILE}")"
        else
            success "Extension built: $(basename "${SO_FILE}")"
        fi
    else
        warn "cv_backend .so not found – check build logs."
    fi
}

# =============================================================================
# Step 3 – Run the application
# =============================================================================
run_app() {
    if [[ ! -x "${PYTHON}" ]]; then
        error "Virtual environment not set up. Run './build.sh' first."
        exit 1
    fi
    info "Launching application …"
    "${PYTHON}" "${SCRIPT_DIR}/main.py"
}

# =============================================================================
# Clean
# =============================================================================
clean() {
    info "Removing build artefacts …"
    rm -rf "${SCRIPT_DIR}/build" "${SCRIPT_DIR}/_skbuild" "${SCRIPT_DIR}/*.egg-info"
    find "${SCRIPT_DIR}" -maxdepth 1 -name "cv_backend*.so" -delete
    success "Clean complete."
}

# =============================================================================
# Main
# =============================================================================
echo -e "${BOLD}=============================="
echo -e " CV Task 4 – Build Script"
echo -e "==============================${NC}"

case "${MODE}" in
    all)
        install_deps
        build_cpp
        run_app
        ;;
    deps)
        install_deps
        ;;
    cpp)
        build_cpp
        ;;
    run) # Deprecated
        install_deps
        build_cpp
        run_app
        ;;
    clean)
        clean
        ;;
esac
