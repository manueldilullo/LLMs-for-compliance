#!/bin/bash
# Setup script for Linux/Mac - LLMs-for-compliance

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}===============================================================${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${BLUE}===============================================================${NC}"
    echo ""
}

print_info() {
    echo -e "${BLUE}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

show_menu() {
    echo ""
    echo "Choose installation type:"
    echo "  1. Auto-detect and install (recommended)"
    echo "  2. CPU-only installation"
    echo "  3. GPU installation (requires CUDA)"
    echo "  4. Install optional packages (Datapizza, Unsloth)"
    echo "  5. Install development tools"
    echo "  6. Check system"
    echo "  7. Test installation"
    echo "  0. Exit"
    echo ""
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found"
        exit 1
    fi
    print_success "Python found: $(python3 --version)"
}

auto_install() {
    print_header "Auto-detect Installation"
    python3 setup.py install
}

cpu_install() {
    print_header "CPU-only Installation"
    python3 setup.py install-cpu
}

gpu_install() {
    print_header "GPU Installation"
    python3 setup.py install-gpu
}

optional_install() {
    print_header "Optional Packages"
    python3 setup.py install-optional
}

dev_install() {
    print_header "Development Tools"
    python3 setup.py install-dev
}

check_system() {
    print_header "System Check"
    python3 setup.py check
}

test_installation() {
    print_header "Testing Installation"
    python3 setup.py test
    
    echo ""
    print_info "Testing custom modules..."
    python3 test_init_llm.py
    python3 test_datapizza_rag.py
}

# Main script
print_header "LLMs-for-compliance - Setup Script"

check_python

# If arguments provided, use them
if [ $# -gt 0 ]; then
    case "$1" in
        install)
            auto_install
            ;;
        install-cpu)
            cpu_install
            ;;
        install-gpu)
            gpu_install
            ;;
        install-optional)
            optional_install
            ;;
        install-dev)
            dev_install
            ;;
        check)
            check_system
            ;;
        test)
            test_installation
            ;;
        *)
            echo "Usage: $0 [install|install-cpu|install-gpu|install-optional|install-dev|check|test]"
            exit 1
            ;;
    esac
    exit 0
fi

# Interactive menu
while true; do
    show_menu
    read -p "Enter choice: " choice
    
    case $choice in
        1)
            auto_install
            ;;
        2)
            cpu_install
            ;;
        3)
            gpu_install
            ;;
        4)
            optional_install
            ;;
        5)
            dev_install
            ;;
        6)
            check_system
            ;;
        7)
            test_installation
            ;;
        0)
            echo ""
            print_success "Setup script finished"
            exit 0
            ;;
        *)
            print_error "Invalid choice"
            ;;
    esac
done
