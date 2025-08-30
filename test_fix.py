#!/usr/bin/env python3
"""
Test script to verify the grayscale conversion fix.
"""

import numpy as np
import sys
import os

# Add the findpeaks directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'findpeaks'))

try:
    from findpeaks import findpeaks
    from findpeaks.stats import togray
    print("✓ Successfully imported findpeaks")
except ImportError as e:
    print(f"✗ Failed to import findpeaks: {e}")
    sys.exit(1)

def test_togray_function():
    """Test the togray function with different input types."""
    print("\n=== Testing togray function ===")
    
    # Test 1: Already grayscale image (2D)
    print("\nTest 1: Already grayscale image (2D)")
    gray_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    print(f"Input shape: {gray_img.shape}, dtype: {gray_img.dtype}")
    
    try:
        result = togray(gray_img)
        print(f"✓ Success: Output shape: {result.shape}, dtype: {result.dtype}")
        assert result.shape == gray_img.shape, "Shape should be preserved"
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 2: Color image (3D)
    print("\nTest 2: Color image (3D)")
    color_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    print(f"Input shape: {color_img.shape}, dtype: {color_img.dtype}")
    
    try:
        result = togray(color_img)
        print(f"✓ Success: Output shape: {result.shape}, dtype: {result.dtype}")
        assert len(result.shape) == 2, "Should be converted to 2D"
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 3: RGBA image (4 channels)
    print("\nTest 3: RGBA image (4 channels)")
    rgba_img = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
    print(f"Input shape: {rgba_img.shape}, dtype: {rgba_img.dtype}")
    
    try:
        result = togray(rgba_img)
        print(f"✓ Success: Output shape: {result.shape}, dtype: {result.dtype}")
        assert len(result.shape) == 2, "Should be converted to 2D"
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 4: Invalid image (1D)
    print("\nTest 4: Invalid image (1D)")
    invalid_img = np.random.randint(0, 255, (100,), dtype=np.uint8)
    print(f"Input shape: {invalid_img.shape}, dtype: {invalid_img.dtype}")
    
    try:
        result = togray(invalid_img)
        print(f"✓ Success: Output shape: {result.shape}, dtype: {result.dtype}")
        assert result.shape == invalid_img.shape, "Should return original for invalid input"
    except Exception as e:
        print(f"✗ Failed: {e}")

def test_findpeaks_integration():
    """Test the findpeaks class with grayscale images."""
    print("\n=== Testing findpeaks integration ===")
    
    # Create a simple grayscale image
    gray_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    print(f"Test image shape: {gray_img.shape}, dtype: {gray_img.dtype}")
    
    try:
        # Test with topology method and togray=True
        fp = findpeaks(method='topology', togray=True, scale=True)
        print("✓ Successfully created findpeaks instance")
        
        # This should not crash with the grayscale image
        results = fp.fit(gray_img)
        print("✓ Successfully processed grayscale image")
        print(f"Results keys: {list(results.keys())}")
        
    except Exception as e:
        print(f"✗ Failed: {e}")

if __name__ == "__main__":
    print("Testing findpeaks grayscale conversion fix...")
    
    test_togray_function()
    test_findpeaks_integration()
    
    print("\n=== Test completed ===")
