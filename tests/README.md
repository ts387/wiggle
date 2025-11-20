# Wiggle Tests

This directory contains unit tests for the Wiggle codebase.

## Running Tests

### Prerequisites

Install pytest:
```bash
/path/to/ChimeraX/bin/python3.9 -m pip install pytest
```

### Run All Tests

```bash
cd /path/to/wiggle
/path/to/ChimeraX/bin/python3.9 -m pytest tests/ -v
```

### Run Specific Test File

```bash
/path/to/ChimeraX/bin/python3.9 -m pytest tests/test_device_utils.py -v
```

### Run Specific Test

```bash
/path/to/ChimeraX/bin/python3.9 -m pytest tests/test_device_utils.py::TestDeviceDetection::test_device_caching -v
```

## Test Coverage

Current test coverage:

- ✅ `test_device_utils.py` - Device detection, memory estimation, safe device operations
- ⚠️ `test_miniDRGN.py` - TODO: Neural network volume generation tests
- ⚠️ `test_miniSPARC.py` - TODO: FFT and component-based rendering tests

## Writing New Tests

When adding new functionality:

1. Create test file: `test_<module_name>.py`
2. Import the module to test
3. Write test classes inheriting from appropriate base
4. Use descriptive test names: `test_<what_it_tests>`
5. Add docstrings explaining what is tested

Example:
```python
class TestNewFeature:
    """Tests for new feature."""

    def test_basic_functionality(self):
        """Test that basic feature works."""
        result = new_feature()
        assert result is not None
```

## Test Guidelines

- **Fast**: Tests should run quickly (< 1s each)
- **Isolated**: Each test should be independent
- **Deterministic**: Same input = same output
- **Clear**: Test names and assertions should be self-documenting
- **Mock GPU**: For GPU tests, provide CPU fallback or skip if unavailable

## Continuous Integration

Currently tests are run manually. Future: GitHub Actions CI/CD.

## Known Limitations

- GPU-specific tests require actual GPU hardware
- MPS tests require M-series Mac
- Some tests may need actual ChimeraX environment (not just Python)
- Model loading tests require test data fixtures (not yet implemented)

## Contributing

When submitting PRs:
1. Add tests for new features
2. Ensure existing tests pass
3. Aim for >80% code coverage on new code
4. Document any test dependencies or fixtures
