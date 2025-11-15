# @Author: Charles Bayly-Jones, Monash University
# @Modified for Metal GPU support
# @Last modified time: 2025-11-15
# @License: GNU General Public License v3.0

"""
Device detection and selection utilities for cross-platform GPU acceleration.
Supports NVIDIA CUDA, Apple Metal (MPS), and CPU fallback.
"""

import torch
import sys


def get_available_device(verbose=True):
    """
    Detect and return the best available compute device.

    Priority order: CUDA > MPS (Metal) > CPU

    Args:
        verbose (bool): Print device information

    Returns:
        tuple: (device, device_type, device_name)
            - device: torch.device object
            - device_type: str ('cuda', 'mps', or 'cpu')
            - device_name: str (human-readable device name)
    """

    # Check for NVIDIA CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_type = 'cuda'
        try:
            device_name = torch.cuda.get_device_name(0)
        except:
            device_name = "NVIDIA GPU"

        if verbose:
            print(f"GPU acceleration: CUDA detected ({device_name})")

        return device, device_type, device_name

    # Check for Apple Metal (MPS)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        device_type = 'mps'

        # Try to detect M-series chip
        try:
            import platform
            import subprocess
            if platform.system() == 'Darwin':  # macOS
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                      capture_output=True, text=True)
                cpu_brand = result.stdout.strip()
                if 'Apple' in cpu_brand:
                    device_name = cpu_brand
                else:
                    device_name = "Apple Metal GPU"
            else:
                device_name = "Apple Metal GPU"
        except:
            device_name = "Apple Metal GPU"

        if verbose:
            print(f"GPU acceleration: Metal (MPS) detected ({device_name})")

        return device, device_type, device_name

    # CPU fallback
    else:
        device = torch.device('cpu')
        device_type = 'cpu'
        device_name = "CPU"

        if verbose:
            print("WARNING: No GPU detected. Using CPU (this will be slow)")
            print("For GPU acceleration:")
            print("  - On NVIDIA systems: Install CUDA toolkit and PyTorch with CUDA support")
            print("  - On M-series Macs: Install PyTorch with MPS support (v1.12+)")

        return device, device_type, device_name


def set_default_device(device):
    """
    Set the default tensor type for the given device.

    Args:
        device (torch.device): The device to set as default
    """
    if device.type == 'cuda':
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    elif device.type == 'mps':
        # MPS doesn't support set_default_tensor_type, so we use set_default_device instead
        # This requires PyTorch 2.0+
        try:
            torch.set_default_device(device)
        except AttributeError:
            # Fallback for older PyTorch versions
            print("WARNING: PyTorch version doesn't support set_default_device for MPS")
            print("         Tensors will need to be manually moved to MPS device")
    else:
        torch.set_default_tensor_type(torch.FloatTensor)


def move_to_device(tensor, device):
    """
    Move a tensor to the specified device with proper error handling.

    Args:
        tensor (torch.Tensor): Input tensor
        device (torch.device): Target device

    Returns:
        torch.Tensor: Tensor on the target device
    """
    try:
        return tensor.to(device)
    except Exception as e:
        print(f"WARNING: Failed to move tensor to {device}: {e}")
        print("         Falling back to CPU")
        return tensor.to('cpu')


def test_device_performance(device, test_size=256):
    """
    Run a simple performance test on the device.

    Args:
        device (torch.device): Device to test
        test_size (int): Size of test tensor

    Returns:
        float: Time in seconds for test operation
    """
    import time

    # Create test tensors
    a = torch.randn(test_size, test_size, device=device)
    b = torch.randn(test_size, test_size, device=device)

    # Warmup
    for _ in range(10):
        c = torch.matmul(a, b)

    # Time the operation
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        # MPS synchronization if available
        try:
            torch.mps.synchronize()
        except:
            pass

    start = time.time()
    for _ in range(100):
        c = torch.matmul(a, b)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        try:
            torch.mps.synchronize()
        except:
            pass

    elapsed = time.time() - start

    return elapsed


def print_device_info():
    """Print detailed information about available compute devices."""
    print("=" * 60)
    print("Wiggle GPU Device Information")
    print("=" * 60)

    device, device_type, device_name = get_available_device(verbose=False)

    print(f"\nSelected Device: {device_type.upper()}")
    print(f"Device Name: {device_name}")
    print(f"PyTorch Version: {torch.__version__}")

    print("\nAvailable Backends:")
    print(f"  CUDA:  {'✓' if torch.cuda.is_available() else '✗'}")
    if torch.cuda.is_available():
        print(f"    - Device: {torch.cuda.get_device_name(0)}")
        print(f"    - CUDA Version: {torch.version.cuda}")

    print(f"  Metal: {'✓' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else '✗'}")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"    - MPS Backend: Available")

    print(f"  CPU:   ✓ (always available)")

    # Performance test
    print("\nRunning quick performance test...")
    try:
        test_time = test_device_performance(device, test_size=128)
        print(f"  Matrix multiplication benchmark: {test_time:.4f}s (100 iterations)")
    except Exception as e:
        print(f"  Performance test failed: {e}")

    print("=" * 60)


if __name__ == "__main__":
    # If run directly, print device information
    print_device_info()
