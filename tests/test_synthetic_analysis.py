# tests/test_synthetic_analysis.py
import pytest
import sys
import os

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from main import synthetic_analysis


def test_synthetic_analysis():
    """
    Test that synthetic_analysis returns valid results.

    This test needs a lot of improvment!
    """
    fixed_correlated_field, fixed_ps = synthetic_analysis()

    # Assert that the results are not None
    assert fixed_correlated_field is not None, "Correlated field should not be None"
    assert fixed_ps is not None, "Power spectrum should not be None"

    # Add more specific assertions based on expected outputs
    # For example, check the shapes of the returned arrays
    expected_shape = (129,)  # Replace with the expected shape
    assert fixed_correlated_field.shape == expected_shape, f"Expected shape {expected_shape}, got {fixed_correlated_field.shape}"