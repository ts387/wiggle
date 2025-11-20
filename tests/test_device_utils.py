"""
Unit tests for device_utils module.

Run with pytest:
    pytest tests/test_device_utils.py

Or run all tests:
    pytest tests/
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from deps import device_utils


class TestDeviceDetection:
    """Tests for device detection functionality."""

    def test_get_available_device_returns_tuple(self):
        """Test that get_available_device returns a 3-tuple."""
        device, device_type, device_name = device_utils.get_available_device(verbose=False)

        assert isinstance(device, torch.device)
        assert isinstance(device_type, str)
        assert isinstance(device_name, str)
        assert device_type in ['cuda', 'mps', 'cpu']

    def test_device_caching(self):
        """Test that device detection is cached."""
        # First call
        device1, type1, name1 = device_utils.get_available_device(verbose=False)

        # Second call should return cached result
        device2, type2, name2 = device_utils.get_available_device(verbose=False)

        assert device1 == device2
        assert type1 == type2
        assert name1 == name2

    def test_force_redetect(self):
        """Test force_redetect parameter."""
        device1, _, _ = device_utils.get_available_device(verbose=False)
        device2, _, _ = device_utils.get_available_device(verbose=False, force_redetect=True)

        # Should still be same device, but re-detected
        assert device1 == device2


class TestMemoryEstimation:
    """Tests for memory estimation functionality."""

    def test_estimate_volume_memory_basic(self):
        """Test basic volume memory estimation."""
        # 256^3 float32 volume with safety factor 4
        estimated = device_utils.estimate_volume_memory(256, dtype_bytes=4, safety_factor=4)

        # 256^3 * 4 bytes * 4 = 268,435,456 bytes
        expected = 256 ** 3 * 4 * 4
        assert estimated == expected

    def test_estimate_volume_memory_complex(self):
        """Test volume memory estimation with complex64."""
        # 128^3 complex64 volume (8 bytes per element)
        estimated = device_utils.estimate_volume_memory(128, dtype_bytes=8, safety_factor=4)

        expected = 128 ** 3 * 8 * 4
        assert estimated == expected

    def test_check_available_memory_returns_tuple(self):
        """Test that memory check returns proper tuple."""
        device = torch.device('cpu')
        available, total = device_utils.check_available_memory(device)

        # Should return tuple (may be None, None if psutil unavailable)
        assert isinstance(available, (int, type(None)))
        assert isinstance(total, (int, type(None)))

        if available is not None:
            assert available >= 0
            assert total >= available

    def test_check_volume_fits_memory_runs(self):
        """Test that volume fit check completes without error."""
        device = torch.device('cpu')

        # Should not raise exception
        fits = device_utils.check_volume_fits_memory(64, device, verbose=False)
        assert isinstance(fits, bool)


class TestSafeToDevice:
    """Tests for safe tensor device movement."""

    def test_safe_to_device_cpu(self):
        """Test safe_to_device with CPU."""
        tensor = torch.randn(10, 10)
        device = torch.device('cpu')

        result = device_utils.safe_to_device(tensor, device, "test_operation")

        assert result.device.type == 'cpu'
        assert torch.allclose(tensor, result)

    def test_safe_to_device_accepts_string(self):
        """Test that safe_to_device accepts device as string."""
        tensor = torch.randn(10, 10)

        result = device_utils.safe_to_device(tensor, 'cpu', "test_operation")

        assert result.device.type == 'cpu'


class TestSetDefaultDevice:
    """Tests for set_default_device functionality."""

    def test_set_default_device_cpu(self):
        """Test setting CPU as default device."""
        device = torch.device('cpu')

        # Should not raise exception
        device_utils.set_default_device(device)

        # Verify we can create tensors
        tensor = torch.randn(5, 5)
        assert tensor is not None


# Integration test
class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_device_workflow(self):
        """Test complete device detection and setup workflow."""
        # Detect device
        device, device_type, device_name = device_utils.get_available_device(verbose=False)

        # Set as default
        device_utils.set_default_device(device)

        # Create tensor
        tensor = torch.randn(10, 10)

        # Move to device safely
        tensor_on_device = device_utils.safe_to_device(tensor, device, "test")

        assert tensor_on_device is not None
        # For CPU and CUDA, check device type
        if device_type in ['cpu', 'cuda']:
            assert tensor_on_device.device.type == device_type


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
