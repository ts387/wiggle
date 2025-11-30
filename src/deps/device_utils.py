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
import warnings


# Global cache for device detection to avoid redundant checks
_CACHED_DEVICE = None


def get_available_device(verbose=True, force_redetect=False):
    """
    Detect and return the best available compute device.

    Priority order: CUDA > MPS (Metal) > CPU

    Args:
        verbose (bool): Print device information
        force_redetect (bool): Force re-detection even if cached result exists

    Returns:
        tuple: (device, device_type, device_name)
            - device: torch.device object
            - device_type: str ('cuda', 'mps', or 'cpu')
            - device_name: str (human-readable device name)
    """
    global _CACHED_DEVICE

    # Return cached result if available (unless force_redetect is True)
    if _CACHED_DEVICE is not None and not force_redetect:
        if verbose:
            device, device_type, device_name = _CACHED_DEVICE
            print(f"Using cached device: {device_type.upper()} ({device_name})")
        return _CACHED_DEVICE

    # Check for NVIDIA CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_type = 'cuda'
        try:
            device_name = torch.cuda.get_device_name(0)
        except (RuntimeError, AssertionError):
            device_name = "NVIDIA GPU"

        if verbose:
            print(f"GPU acceleration: CUDA detected ({device_name})")

        _CACHED_DEVICE = (device, device_type, device_name)
        return _CACHED_DEVICE

    # Check for Apple Metal (MPS)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        device_type = 'mps'

        # Check PyTorch version for MPS compatibility
        torch_version = torch.__version__.split('+')[0]
        try:
            major, minor = map(int, torch_version.split('.')[:2])
        except (ValueError, IndexError):
            major, minor = 1, 0

        # Try to detect M-series chip
        try:
            import platform
            import subprocess
            if platform.system() == 'Darwin':  # macOS
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                      capture_output=True, text=True, timeout=5)
                cpu_brand = result.stdout.strip()
                if 'Apple' in cpu_brand:
                    device_name = cpu_brand
                else:
                    device_name = "Apple Metal GPU"
            else:
                device_name = "Apple Metal GPU"
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError, FileNotFoundError):
            device_name = "Apple Metal GPU"

        # Check for PyTorch version compatibility (must check before verbose output)
        # Check the more specific case first (too old for MPS)
        if major == 1 and minor < 12:
            if verbose:
                print(f"WARNING: PyTorch {torch_version} too old for MPS support (requires 1.12+)")
                print("         Falling back to CPU")
            device = torch.device('cpu')
            device_type = 'cpu'
            device_name = "CPU"
        else:
            if verbose:
                print(f"GPU acceleration: Metal (MPS) detected ({device_name})")
                # Warn about PyTorch version compatibility
                if major < 2:
                    print(f"WARNING: PyTorch {torch_version} detected. For best MPS compatibility, upgrade to PyTorch 2.0+")
                    print("         Some operations may fall back to CPU")

        _CACHED_DEVICE = (device, device_type, device_name)
        return _CACHED_DEVICE

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

        _CACHED_DEVICE = (device, device_type, device_name)
        return _CACHED_DEVICE


def set_default_device(device):
    """
    Set the default tensor type for the given device with robust error handling.

    Args:
        device (torch.device): The device to set as default
    """
    try:
        # Prefer the modern set_default_device API (PyTorch 2.0+) for all device types
        if hasattr(torch, 'set_default_device'):
            try:
                torch.set_default_device(device)
                return
            except RuntimeError as e:
                # Some devices may not be fully supported, fall through to legacy API
                if device.type not in ('cuda', 'cpu'):
                    warnings.warn(
                        f"Failed to set {device.type} as default device: {e}. "
                        "Tensors will be created on CPU and moved as needed.",
                        RuntimeWarning
                    )
                    return

        # Legacy fallback for older PyTorch versions or if set_default_device fails
        if device.type == 'cuda':
            # Use deprecated API only for older PyTorch versions
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=DeprecationWarning)
                torch.set_default_tensor_type(torch.cuda.FloatTensor)
        elif device.type == 'mps':
            # MPS doesn't support set_default_tensor_type
            warnings.warn(
                "PyTorch version doesn't support set_default_device for MPS. "
                "Tensors will need to be manually moved to MPS device. "
                "Consider upgrading to PyTorch 2.0+",
                RuntimeWarning
            )
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=DeprecationWarning)
                torch.set_default_tensor_type(torch.FloatTensor)

    except RuntimeError as e:
        # CUDA initialization can fail for various reasons
        warnings.warn(
            f"Failed to set default device to {device.type}: {e}\n"
            f"Continuing with CPU tensors. Device operations will explicitly move tensors as needed.",
            RuntimeWarning
        )
    except Exception as e:
        warnings.warn(
            f"Unexpected error setting default device: {e}\n"
            f"Falling back to CPU default tensors.",
            RuntimeWarning
        )


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


def safe_to_device(tensor, device, operation_name="operation"):
    """
    Safely move tensor to device with MPS compatibility fallback.

    This function provides robust error handling for MPS operations that may
    not be fully supported in all PyTorch versions.

    Args:
        tensor (torch.Tensor): Input tensor
        device (torch.device or str): Target device
        operation_name (str): Name of the operation for error reporting

    Returns:
        torch.Tensor: Tensor on the target device (or CPU if MPS fails)
    """
    if isinstance(device, str):
        device = torch.device(device)

    try:
        return tensor.to(device)
    except (RuntimeError, NotImplementedError) as e:
        if device.type == 'mps':
            warnings.warn(
                f"MPS operation '{operation_name}' not supported: {e}\n"
                f"Falling back to CPU for this operation. "
                f"Consider upgrading PyTorch to version 2.0+ for better MPS support.",
                RuntimeWarning
            )
            return tensor.to('cpu')
        else:
            # For non-MPS devices, re-raise the exception
            raise
    except Exception as e:
        # Catch-all for unexpected errors
        warnings.warn(
            f"Unexpected error in '{operation_name}' on {device.type}: {e}\n"
            f"Falling back to CPU.",
            RuntimeWarning
        )
        return tensor.to('cpu')


def estimate_volume_memory(volume_size, dtype_bytes=4, safety_factor=4):
    """
    Estimate GPU memory required for volume processing.

    Args:
        volume_size (int): Size of cubic volume (e.g., 256 for 256³)
        dtype_bytes (int): Bytes per element (4 for float32, 8 for complex64)
        safety_factor (int): Multiplier for intermediate tensors (default 4)

    Returns:
        int: Estimated memory in bytes
    """
    # Base volume size
    base_memory = volume_size ** 3 * dtype_bytes

    # Account for FFT intermediate buffers, gradients, etc.
    estimated_total = base_memory * safety_factor

    return estimated_total


def check_available_memory(device):
    """
    Check available memory on the device.

    Args:
        device (torch.device): Device to check

    Returns:
        tuple: (available_bytes, total_bytes) or (None, None) if unavailable
    """
    try:
        if device.type == 'cuda':
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            available = total - max(allocated, reserved)
            return available, total

        elif device.type == 'mps':
            # MPS uses unified memory - check system memory
            try:
                import psutil
                mem = psutil.virtual_memory()
                return mem.available, mem.total
            except ImportError:
                warnings.warn(
                    "psutil not installed - cannot check memory. "
                    "Install with: pip install psutil",
                    RuntimeWarning
                )
                return None, None

        else:  # CPU
            try:
                import psutil
                mem = psutil.virtual_memory()
                return mem.available, mem.total
            except ImportError:
                return None, None

    except Exception as e:
        warnings.warn(f"Failed to check memory: {e}", RuntimeWarning)
        return None, None


def check_volume_fits_memory(volume_size, device, verbose=True):
    """
    Check if a volume of given size will fit in device memory.

    Args:
        volume_size (int): Size of cubic volume
        device (torch.device): Target device
        verbose (bool): Print warnings

    Returns:
        bool: True if volume should fit, False if it likely won't
    """
    required = estimate_volume_memory(volume_size)
    available, total = check_available_memory(device)

    if available is None:
        # Can't check - assume it fits
        if verbose:
            print(f"INFO: Unable to verify memory for {volume_size}³ volume")
        return True

    fits = available > required

    if verbose:
        required_mb = required / (1024 ** 2)
        available_mb = available / (1024 ** 2)
        total_mb = total / (1024 ** 2)

        if fits:
            print(f"INFO: Volume {volume_size}³ requires ~{required_mb:.0f}MB, "
                  f"{available_mb:.0f}MB available ({total_mb:.0f}MB total)")
        else:
            print(f"WARNING: Volume {volume_size}³ requires ~{required_mb:.0f}MB, "
                  f"but only {available_mb:.0f}MB available ({total_mb:.0f}MB total)")
            print(f"         Consider:")
            print(f"         - Reducing volume size via downsampling")
            print(f"         - Cropping to remove empty regions")
            print(f"         - Using CPU mode (slower but more memory)")

    return fits


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
        except (AttributeError, RuntimeError):
            pass

    start = time.time()
    for _ in range(100):
        c = torch.matmul(a, b)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        try:
            torch.mps.synchronize()
        except (AttributeError, RuntimeError):
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
