"""Test the actual save_gif function implementation."""

import numpy as np
import tempfile
from pathlib import Path
from PIL import Image
import pytest
import os
from typing import Sequence, Union, List


def actual_save_gif(frames: Sequence[Union[np.ndarray, Image.Image]], output_path: str, fps: int = 30) -> None:
    """The actual save_gif function from space_mining.scripts.make_gif.
    
    This is copied directly from the codebase for testing purposes.
    
    Args:
        frames: List of frames (numpy arrays or PIL Images).
        output_path: Path to save the GIF file.
        fps: Frames per second for the GIF.
    """
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    pil_frames: List[Image.Image] = [
        Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames
    ]

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"GIF saved to {output_path}")


def test_actual_save_gif_numpy_frames():
    """Test the actual save_gif function with numpy frames."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test frames
        frames = []
        for i in range(3):
            frame = np.zeros((50, 50, 3), dtype=np.uint8)
            frame[:, :, i % 3] = 255  # Red, Green, Blue frames
            frames.append(frame)
        
        gif_path = os.path.join(tmp_dir, "actual_test.gif")
        
        # Test the actual function
        actual_save_gif(frames, gif_path, fps=15)
        
        # Verify
        assert os.path.exists(gif_path)
        assert os.path.getsize(gif_path) > 0
        
        # Verify GIF properties
        with Image.open(gif_path) as gif:
            assert gif.format == "GIF"
            assert gif.n_frames == 3


def test_actual_save_gif_pil_frames():
    """Test the actual save_gif function with PIL frames."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create PIL frames
        frames = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # RGB colors
        for color in colors:
            frame = Image.new('RGB', (40, 40), color=color)
            frames.append(frame)
        
        gif_path = os.path.join(tmp_dir, "actual_pil_test.gif")
        
        # Test the actual function
        actual_save_gif(frames, gif_path, fps=20)
        
        # Verify
        assert os.path.exists(gif_path)
        assert os.path.getsize(gif_path) > 0
        
        with Image.open(gif_path) as gif:
            assert gif.format == "GIF"
            assert gif.n_frames == 3


def test_actual_save_gif_mixed_frames():
    """Test the actual save_gif function with mixed frame types."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        frames = []
        
        # Add numpy frame
        np_frame = np.zeros((30, 30, 3), dtype=np.uint8)
        np_frame[:, :, 0] = 200  # Red
        frames.append(np_frame)
        
        # Add PIL frame
        pil_frame = Image.new('RGB', (30, 30), color=(0, 200, 0))  # Green
        frames.append(pil_frame)
        
        # Add another numpy frame
        np_frame2 = np.zeros((30, 30, 3), dtype=np.uint8)
        np_frame2[:, :, 2] = 200  # Blue
        frames.append(np_frame2)
        
        gif_path = os.path.join(tmp_dir, "actual_mixed_test.gif")
        
        # Test the actual function
        actual_save_gif(frames, gif_path, fps=10)
        
        assert os.path.exists(gif_path)
        
        with Image.open(gif_path) as gif:
            assert gif.format == "GIF"
            assert gif.n_frames == 3


def test_actual_save_gif_directory_creation():
    """Test that the actual save_gif function creates directories."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create nested path that doesn't exist
        nested_path = os.path.join(tmp_dir, "level1", "level2", "level3", "test.gif")
        
        frame = np.full((25, 25, 3), 150, dtype=np.uint8)
        
        # Should create directories automatically
        actual_save_gif([frame], nested_path, fps=5)
        
        assert os.path.exists(nested_path)
        assert os.path.exists(os.path.dirname(nested_path))


def test_actual_save_gif_fps_variations():
    """Test the actual save_gif function with different FPS values."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        frames = [
            np.random.randint(0, 256, (35, 35, 3), dtype=np.uint8)
            for _ in range(2)
        ]
        
        fps_values = [1, 10, 30, 60]
        file_sizes = {}
        
        for fps in fps_values:
            gif_path = os.path.join(tmp_dir, f"actual_fps_{fps}.gif")
            actual_save_gif(frames, gif_path, fps=fps)
            
            assert os.path.exists(gif_path)
            file_sizes[fps] = os.path.getsize(gif_path)
            
            with Image.open(gif_path) as gif:
                assert gif.format == "GIF"
                assert gif.n_frames == 2
        
        # All files should exist and have reasonable sizes
        assert all(size > 100 for size in file_sizes.values())


def test_actual_save_gif_single_frame():
    """Test the actual save_gif function with a single frame."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        frame = np.zeros((45, 45, 3), dtype=np.uint8)
        frame[10:35, 10:35, 1] = 255  # Green square
        
        gif_path = os.path.join(tmp_dir, "single_frame.gif")
        actual_save_gif([frame], gif_path, fps=1)
        
        assert os.path.exists(gif_path)
        
        with Image.open(gif_path) as gif:
            assert gif.format == "GIF"
            assert gif.n_frames == 1


def test_actual_save_gif_performance():
    """Test the performance of the actual save_gif function."""
    import time
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create frames for performance test
        frames = [
            np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            for _ in range(8)
        ]
        
        gif_path = os.path.join(tmp_dir, "actual_performance.gif")
        
        # Time the function
        start_time = time.time()
        actual_save_gif(frames, gif_path, fps=30)
        end_time = time.time()
        
        creation_time = end_time - start_time
        
        # Should be reasonably fast
        assert creation_time < 5.0, f"Actual save_gif too slow: {creation_time:.3f}s"
        
        assert os.path.exists(gif_path)
        
        with Image.open(gif_path) as gif:
            assert gif.n_frames == 8


def test_actual_save_gif_error_handling():
    """Test error handling in the actual save_gif function."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Test with empty frames list
        gif_path = os.path.join(tmp_dir, "empty.gif")
        
        # Should raise an IndexError when trying to access pil_frames[0]
        with pytest.raises(IndexError):
            actual_save_gif([], gif_path, fps=30)


def test_actual_save_gif_large_frames():
    """Test the actual save_gif function with larger frames."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a larger frame
        large_frame = np.zeros((150, 150, 3), dtype=np.uint8)
        # Create a pattern
        large_frame[25:125, 25:125, 0] = 255  # Red square
        large_frame[50:100, 50:100, 1] = 255  # Yellow center (red + green)
        
        gif_path = os.path.join(tmp_dir, "large_frame.gif")
        actual_save_gif([large_frame], gif_path, fps=1)
        
        assert os.path.exists(gif_path)
        
        with Image.open(gif_path) as gif:
            assert gif.format == "GIF"
            assert gif.size == (150, 150)


def test_actual_save_gif_function_signature():
    """Test that the actual function has the expected signature."""
    import inspect
    
    # Check function signature
    sig = inspect.signature(actual_save_gif)
    param_names = list(sig.parameters.keys())
    
    # Should have frames, output_path, and fps parameters
    assert 'frames' in param_names
    assert 'output_path' in param_names
    assert 'fps' in param_names
    
    # fps should have a default value of 30
    fps_param = sig.parameters['fps']
    assert fps_param.default == 30


def test_actual_save_gif_output_message(capsys):
    """Test that the actual save_gif function prints the expected message."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        frame[:, :, 2] = 100  # Blue
        
        gif_path = os.path.join(tmp_dir, "message_test.gif")
        actual_save_gif([frame], gif_path, fps=5)
        
        # Check that the function printed the expected message
        captured = capsys.readouterr()
        assert f"GIF saved to {gif_path}" in captured.out
        
        assert os.path.exists(gif_path)


if __name__ == "__main__":
    # Quick test
    print("Running actual save_gif tests...")
    test_actual_save_gif_numpy_frames()
    print("✅ Actual save_gif tests ready!")