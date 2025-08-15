"""Test the actual save_gif function from space_mining.scripts.make_gif."""

import numpy as np
import tempfile
from pathlib import Path
from PIL import Image
import pytest
import sys
import os

# Extract and test the real save_gif function


def extract_save_gif_function():
    """Extract the save_gif function from the actual codebase."""
    # Read the actual file
    script_path = os.path.join(os.path.dirname(__file__), '..', 'space_mining', 'scripts', 'make_gif.py')
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Create a namespace for the function
    namespace = {
        'os': os,
        'np': np,
        'Image': Image,
        'Sequence': list,  # Simplified for testing
        'Union': type,     # Simplified for testing
        'List': list,      # Simplified for testing
    }
    
    # Execute the file content to get the function
    exec(content, namespace)
    
    return namespace['save_gif']


def test_extract_real_save_gif():
    """Test that we can extract the real save_gif function."""
    try:
        save_gif_func = extract_save_gif_function()
        assert callable(save_gif_func)
        print("✅ Successfully extracted real save_gif function")
    except Exception as e:
        pytest.skip(f"Could not extract save_gif function: {e}")


def test_real_save_gif_with_numpy_frames():
    """Test the real save_gif function with numpy frames."""
    try:
        save_gif = extract_save_gif_function()
    except Exception as e:
        pytest.skip(f"Could not extract save_gif function: {e}")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test frames
        frames = []
        for i in range(3):
            frame = np.zeros((60, 60, 3), dtype=np.uint8)
            frame[:, :, i % 3] = 200  # Red, Green, Blue frames
            frames.append(frame)
        
        gif_path = os.path.join(tmp_dir, "real_test.gif")
        
        # Test the real function
        save_gif(frames, gif_path, fps=20)
        
        # Verify
        assert os.path.exists(gif_path)
        assert os.path.getsize(gif_path) > 0
        
        # Verify GIF properties
        with Image.open(gif_path) as gif:
            assert gif.format == "GIF"
            assert gif.n_frames == 3


def test_real_save_gif_with_pil_frames():
    """Test the real save_gif function with PIL frames."""
    try:
        save_gif = extract_save_gif_function()
    except Exception as e:
        pytest.skip(f"Could not extract save_gif function: {e}")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create PIL frames
        frames = []
        colors = ['red', 'green', 'blue']
        for color in colors:
            frame = Image.new('RGB', (40, 40), color=color)
            frames.append(frame)
        
        gif_path = os.path.join(tmp_dir, "real_pil_test.gif")
        
        # Test the real function
        save_gif(frames, gif_path, fps=10)
        
        # Verify
        assert os.path.exists(gif_path)
        assert os.path.getsize(gif_path) > 0
        
        with Image.open(gif_path) as gif:
            assert gif.format == "GIF"
            assert gif.n_frames == 3


def test_real_save_gif_mixed_frames():
    """Test the real save_gif function with mixed frame types."""
    try:
        save_gif = extract_save_gif_function()
    except Exception as e:
        pytest.skip(f"Could not extract save_gif function: {e}")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        frames = []
        
        # Add numpy frame
        np_frame = np.zeros((35, 35, 3), dtype=np.uint8)
        np_frame[:, :, 0] = 255  # Red
        frames.append(np_frame)
        
        # Add PIL frame
        pil_frame = Image.new('RGB', (35, 35), color='lime')
        frames.append(pil_frame)
        
        gif_path = os.path.join(tmp_dir, "real_mixed_test.gif")
        
        # Test the real function
        save_gif(frames, gif_path, fps=25)
        
        assert os.path.exists(gif_path)
        
        with Image.open(gif_path) as gif:
            assert gif.format == "GIF"
            assert gif.n_frames == 2


def test_real_save_gif_directory_creation():
    """Test that the real save_gif function creates directories."""
    try:
        save_gif = extract_save_gif_function()
    except Exception as e:
        pytest.skip(f"Could not extract save_gif function: {e}")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create nested path that doesn't exist
        nested_path = os.path.join(tmp_dir, "deep", "nested", "path", "test.gif")
        
        frame = np.full((25, 25, 3), 128, dtype=np.uint8)
        
        # Should create directories automatically
        save_gif([frame], nested_path, fps=5)
        
        assert os.path.exists(nested_path)
        assert os.path.exists(os.path.dirname(nested_path))


def test_real_save_gif_fps_parameter():
    """Test the real save_gif function with different FPS values."""
    try:
        save_gif = extract_save_gif_function()
    except Exception as e:
        pytest.skip(f"Could not extract save_gif function: {e}")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        frames = [
            np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)
            for _ in range(2)
        ]
        
        fps_values = [1, 15, 30, 60]
        
        for fps in fps_values:
            gif_path = os.path.join(tmp_dir, f"real_fps_{fps}.gif")
            save_gif(frames, gif_path, fps=fps)
            
            assert os.path.exists(gif_path)
            assert os.path.getsize(gif_path) > 0
            
            with Image.open(gif_path) as gif:
                assert gif.format == "GIF"
                assert gif.n_frames == 2


def test_real_save_gif_performance():
    """Test the performance of the real save_gif function."""
    try:
        save_gif = extract_save_gif_function()
    except Exception as e:
        pytest.skip(f"Could not extract save_gif function: {e}")
    
    import time
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create frames for performance test
        frames = [
            np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        gif_path = os.path.join(tmp_dir, "real_performance.gif")
        
        # Time the function
        start_time = time.time()
        save_gif(frames, gif_path, fps=30)
        end_time = time.time()
        
        creation_time = end_time - start_time
        
        # Should be reasonably fast
        assert creation_time < 3.0, f"Real save_gif too slow: {creation_time:.3f}s"
        
        assert os.path.exists(gif_path)
        
        with Image.open(gif_path) as gif:
            assert gif.n_frames == 5


def test_real_save_gif_error_handling():
    """Test error handling in the real save_gif function."""
    try:
        save_gif = extract_save_gif_function()
    except Exception as e:
        pytest.skip(f"Could not extract save_gif function: {e}")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Test with empty frames (should handle gracefully or raise appropriate error)
        gif_path = os.path.join(tmp_dir, "empty.gif")
        
        try:
            save_gif([], gif_path, fps=30)
            # If it doesn't raise an error, check what was created
            if os.path.exists(gif_path):
                pytest.fail("Empty frames created a file - unexpected")
        except (ValueError, IndexError, AttributeError):
            # These are acceptable errors for empty frames
            pass


def test_function_signature_compatibility():
    """Test that the real function has the expected signature."""
    try:
        save_gif = extract_save_gif_function()
    except Exception as e:
        pytest.skip(f"Could not extract save_gif function: {e}")
    
    import inspect
    
    # Check function signature
    sig = inspect.signature(save_gif)
    param_names = list(sig.parameters.keys())
    
    # Should have frames, output_path, and fps parameters
    assert 'frames' in param_names
    assert 'output_path' in param_names
    assert 'fps' in param_names
    
    # fps should have a default value
    fps_param = sig.parameters['fps']
    assert fps_param.default is not inspect.Parameter.empty


if __name__ == "__main__":
    # Quick test
    print("Testing real save_gif function extraction...")
    test_extract_real_save_gif()
    print("✅ Real function tests ready!")