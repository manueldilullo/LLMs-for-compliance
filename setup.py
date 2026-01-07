#!/usr/bin/env python3
"""
Setup script for LLMs-for-compliance project
Cross-platform alternative to Makefile
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path

# Colors for terminal output
class Colors:
    if platform.system() == "Windows":
        # Windows doesn't support ANSI colors by default in older versions
        try:
            import colorama
            colorama.init()
            BLUE = '\033[0;34m'
            GREEN = '\033[0;32m'
            YELLOW = '\033[1;33m'
            RED = '\033[0;31m'
            NC = '\033[0m'
        except ImportError:
            BLUE = GREEN = YELLOW = RED = NC = ''
    else:
        BLUE = '\033[0;34m'
        GREEN = '\033[0;32m'
        YELLOW = '\033[1;33m'
        RED = '\033[0;31m'
        NC = '\033[0m'

def print_header(text):
    """Print colored header"""
    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.NC}")
    print(f"{Colors.GREEN}{text}{Colors.NC}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.NC}\n")

def print_info(text):
    """Print info message"""
    print(f"{Colors.BLUE}{text}{Colors.NC}")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.NC}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.NC}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.NC}")

def run_command(cmd, check=True, capture_output=False, verbose=False):
    """Run shell command"""
    if verbose:
        print_info(f"Running: {cmd}")
    
    try:
        if capture_output:
            result = subprocess.run(
                cmd,
                shell=True,
                check=check,
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=True, check=check)
            return True
    except subprocess.CalledProcessError as e:
        if verbose:
            print_error(f"Command failed: {e}")
        return None if capture_output else False

def check_python_version():
    """Check if Python version is adequate"""
    print_info("Checking Python version...")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error("Python 3.8+ is required")
        return False
    
    print_success("Python version OK")
    return True

def detect_cuda():
    """Detect CUDA installation"""
    print_info("Checking for CUDA...")
    
    # Try nvcc
    nvcc_version = run_command("nvcc --version", check=False, capture_output=True)
    if nvcc_version and "release" in nvcc_version:
        version = nvcc_version.split("release")[1].split(",")[0].strip()
        print_success(f"CUDA {version} found")
        return version
    
    # Try nvidia-smi
    nvidia_smi = run_command("nvidia-smi", check=False, capture_output=True)
    if nvidia_smi:
        print_warning("nvidia-smi found but nvcc not in PATH")
        return "unknown"
    
    print_warning("CUDA not detected")
    return None

def check_gpu():
    """Check GPU availability via PyTorch"""
    print_info("Checking GPU availability...")
    
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print_success(f"CUDA available: {torch.version.cuda}")
            print(f"  GPU devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    - Device {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print_warning("CUDA not available in PyTorch")
            return False
    except ImportError:
        print_warning("PyTorch not installed yet")
        return None

def install_base_requirements():
    """Install base requirements"""
    print_header("Installing Base Dependencies")
    
    # Upgrade pip
    print_info("Upgrading pip, setuptools, wheel...")
    run_command(f"{sys.executable} -m pip install --upgrade pip setuptools wheel")
    
    # Install from requirements.txt
    print_info("Installing requirements.txt...")
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt"):
        print_error("Failed to install requirements")
        return False
    
    print_success("Base dependencies installed")
    return True

def install_faiss_gpu(cuda_version):
    """Install FAISS with GPU support"""
    print_info("Installing FAISS-GPU...")
    
    if cuda_version and cuda_version.startswith("12"):
        package = "faiss-gpu-cu12"
    else:
        package = "faiss-gpu"
    
    if run_command(f"{sys.executable} -m pip install {package}"):
        print_success(f"FAISS-GPU installed ({package})")
        return True
    else:
        print_error("Failed to install FAISS-GPU")
        return False

def install_faiss_cpu():
    """Install FAISS CPU-only"""
    print_info("Installing FAISS-CPU...")
    
    if run_command(f"{sys.executable} -m pip install faiss-cpu"):
        print_success("FAISS-CPU installed")
        return True
    else:
        print_error("Failed to install FAISS-CPU")
        return False

def install_llama_cpp_gpu():
    """Install llama-cpp-python with GPU support"""
    print_info("Installing llama-cpp-python with CUDA support...")
    print_warning("This may take several minutes...")
    
    if platform.system() == "Windows":
        # Windows command
        cmd = (
            f'set CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native" && '
            f'set FORCE_CMAKE=1 && '
            f'{sys.executable} -m pip install llama-cpp-python --force-reinstall --no-cache-dir --verbose'
        )
    else:
        # Linux/Mac command
        cmd = (
            f'CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native" '
            f'FORCE_CMAKE=1 '
            f'{sys.executable} -m pip install llama-cpp-python --force-reinstall --no-cache-dir --verbose'
        )
    
    if run_command(cmd, verbose=True):
        print_success("llama-cpp-python with CUDA installed")
        return True
    else:
        print_error("Failed to install llama-cpp-python with CUDA")
        return False

def install_llama_cpp_cpu():
    """Install llama-cpp-python CPU-only"""
    print_info("Installing llama-cpp-python (CPU)...")
    
    if run_command(f"{sys.executable} -m pip install llama-cpp-python"):
        print_success("llama-cpp-python (CPU) installed")
        return True
    else:
        print_error("Failed to install llama-cpp-python")
        return False

def install_datapizza():
    """Install Datapizza AI"""
    print_info("Installing Datapizza AI...")
    
    packages = [
        "datapizza-ai",
        "datapizza-ai-parsers-docling",
        "datapizza-ai-embedders-fastembedder",
        "qdrant-client"
    ]
    
    for pkg in packages:
        if not run_command(f"{sys.executable} -m pip install {pkg}"):
            print_warning(f"Failed to install {pkg}")
            return False
    
    print_success("Datapizza AI installed")
    return True

def install_unsloth():
    """Install Unsloth for efficient fine-tuning"""
    print_info("Installing Unsloth...")
    
    cmd = f'{sys.executable} -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"'
    
    if run_command(cmd):
        print_success("Unsloth installed")
        return True
    else:
        print_warning("Failed to install Unsloth (optional)")
        return False

def install_dev_tools():
    """Install development tools"""
    print_info("Installing development tools...")
    
    tools = [
        "jupyter", "ipykernel", "matplotlib", "seaborn",
        "pytest", "black", "flake8", "mypy"
    ]
    
    cmd = f"{sys.executable} -m pip install {' '.join(tools)}"
    
    if run_command(cmd):
        print_success("Development tools installed")
        return True
    else:
        print_warning("Failed to install some dev tools")
        return False

def test_imports():
    """Test core imports"""
    print_header("Testing Core Imports")
    
    modules = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("sentence_transformers", "SentenceTransformers"),
        ("faiss", "FAISS"),
        ("llama_cpp", "llama-cpp-python"),
        ("networkx", "NetworkX"),
        ("pydantic", "Pydantic"),
    ]
    
    success = True
    for module, name in modules:
        try:
            __import__(module)
            print_success(f"{name}")
        except ImportError:
            print_error(f"{name} - NOT FOUND")
            success = False
    
    return success

def test_custom_modules():
    """Test custom module imports"""
    print_header("Testing Custom Modules")
    
    try:
        from retrieval_augmented_generation import RAGFactory
        print_success("retrieval_augmented_generation")
    except ImportError as e:
        print_error(f"retrieval_augmented_generation: {e}")
        return False
    
    try:
        from finetuning import load_model
        print_success("finetuning")
    except ImportError as e:
        print_error(f"finetuning: {e}")
        return False
    
    try:
        from synthetic_dataset_creation import AsyncQAGDPRPipeline
        print_success("synthetic_dataset_creation")
    except ImportError as e:
        print_error(f"synthetic_dataset_creation: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Setup script for LLMs-for-compliance project",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "action",
        choices=[
            "install", "install-cpu", "install-gpu",
            "install-dev", "install-optional",
            "check", "test", "test-gpu", "info"
        ],
        help="Action to perform"
    )
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print_header("LLMs-for-compliance Setup")
    
    # Check Python version first
    if not check_python_version():
        sys.exit(1)
    
    if args.action == "check":
        cuda_version = detect_cuda()
        check_gpu()
        
    elif args.action == "info":
        print_info("Project Structure:")
        print("  • finetuning/                 - Model fine-tuning")
        print("  • synthetic_dataset_creation/ - Dataset generation")
        print("  • retrieval_augmented_generation/ - RAG systems")
        print("  • data/                       - GDPR & AI Act data")
        print("  • notebooks/                  - Jupyter notebooks")
        print("\nMain Scripts:")
        print("  • python -m finetuning.finetuning_with_unsloth")
        print("  • python -m synthetic_dataset_creation.synthetic_data_generation")
        print("  • python -m retrieval_augmented_generation.rag_and_qa")
        
    elif args.action == "install":
        if not install_base_requirements():
            sys.exit(1)
        
        cuda_version = detect_cuda()
        if cuda_version:
            print_info("GPU detected! Installing GPU-specific packages...")
            install_faiss_gpu(cuda_version)
            install_llama_cpp_gpu()
        else:
            print_info("No GPU detected. Installing CPU-only packages...")
            install_faiss_cpu()
            install_llama_cpp_cpu()
        
        print_success("Installation complete!")
        
    elif args.action == "install-cpu":
        if not install_base_requirements():
            sys.exit(1)
        install_faiss_cpu()
        install_llama_cpp_cpu()
        print_success("CPU installation complete!")
        
    elif args.action == "install-gpu":
        if not install_base_requirements():
            sys.exit(1)
        cuda_version = detect_cuda()
        if not cuda_version:
            print_error("CUDA not found! Install CUDA first or use 'install-cpu'")
            sys.exit(1)
        install_faiss_gpu(cuda_version)
        install_llama_cpp_gpu()
        print_success("GPU installation complete!")
        
    elif args.action == "install-dev":
        install_dev_tools()
        
    elif args.action == "install-optional":
        install_datapizza()
        install_unsloth()
        
    elif args.action == "test":
        if not test_imports():
            sys.exit(1)
        print_success("All core imports successful!")
        
    elif args.action == "test-gpu":
        if not check_gpu():
            print_error("GPU not available")
            sys.exit(1)

if __name__ == "__main__":
    main()
