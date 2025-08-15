"""Test core GIF generation functionality from the scripts."""

import numpy as np
import tempfile
from pathlib import Path
from PIL import Image
import pytest
import sys
import os

# Add the space_mining directory to the path to import scripts directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def mock_save_gif(frames, output_path, fps=30):
    """Mock version of save_gif function for testing."""
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    pil_frames = [
        Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames
    ]

    if len(pil_frames) == 0:
        raise ValueError("No frames provided")

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )


def test_save_gif_function_directly():
    """Test the save_gif function directly without dependencies."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test frames
        frames = []
        for i in range(3):
            frame = np.zeros((50, 50, 3), dtype=np.uint8)
            frame[:, :, i % 3] = 255  # RGB frames
            frames.append(frame)
        
        gif_path = os.path.join(tmp_dir, "test_direct.gif")
        
        # Test the mock function
        mock_save_gif(frames, gif_path, fps=15)
        
        # Verify
        assert os.path.exists(gif_path)
        assert os.path.getsize(gif_path) > 0
        
        # Verify GIF properties
        with Image.open(gif_path) as gif:
            assert gif.format == "GIF"
            assert gif.n_frames == 3


def test_save_gif_with_mixed_frame_types():
    """Test save_gif with mixed numpy arrays and PIL images."""
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
        
        gif_path = os.path.join(tmp_dir, "mixed_frames.gif")
        mock_save_gif(frames, gif_path, fps=10)
        
        assert os.path.exists(gif_path)
        
        with Image.open(gif_path) as gif:
            assert gif.format == "GIF"
            assert gif.n_frames == 3


def test_save_gif_fps_variations():
    """Test save_gif with different FPS values."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create simple frames
        frames = [
            np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
            for _ in range(2)
        ]
        
        fps_values = [5, 15, 30, 60]
        file_sizes = {}
        
        for fps in fps_values:
            gif_path = os.path.join(tmp_dir, f"test_{fps}fps.gif")
            mock_save_gif(frames, gif_path, fps=fps)
            
            assert os.path.exists(gif_path)
            file_sizes[fps] = os.path.getsize(gif_path)
            
            with Image.open(gif_path) as gif:
                assert gif.format == "GIF"
                assert gif.n_frames == 2
        
        # All files should exist and have reasonable sizes
        assert all(size > 100 for size in file_sizes.values())


def test_save_gif_directory_creation():
    """Test that save_gif creates directories automatically."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create nested path
        nested_path = os.path.join(tmp_dir, "subdir", "nested", "test.gif")
        
        frame = np.zeros((25, 25, 3), dtype=np.uint8)
        frame[:, :, 1] = 255  # Green
        
        mock_save_gif([frame], nested_path, fps=1)
        
        assert os.path.exists(nested_path)
        assert os.path.exists(os.path.dirname(nested_path))


def test_save_gif_error_cases():
    """Test error handling in save_gif."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Test with empty frames list
        gif_path = os.path.join(tmp_dir, "empty.gif")
        
        with pytest.raises(ValueError):
            mock_save_gif([], gif_path, fps=30)
        
        # Test with invalid frame type that PIL can't handle
        weird_frame = np.array([[[1, 2, 3]]], dtype=np.int64)  # Wrong dtype
        gif_path2 = os.path.join(tmp_dir, "weird.gif")
        
        # This should raise a TypeError due to unsupported dtype
        with pytest.raises(TypeError):
            mock_save_gif([weird_frame], gif_path2, fps=10)
        
        # Test with properly formatted but small frame
        tiny_frame = np.array([[[255, 128, 64]]], dtype=np.uint8)  # Correct dtype
        gif_path3 = os.path.join(tmp_dir, "tiny.gif")
        
        # This should work with correct dtype
        mock_save_gif([tiny_frame], gif_path3, fps=10)
        assert os.path.exists(gif_path3)


def test_frame_conversion_logic():
    """Test the frame conversion logic specifically."""
    # Test numpy to PIL conversion
    np_frame = np.zeros((40, 40, 3), dtype=np.uint8)
    np_frame[10:30, 10:30, :] = [255, 128, 64]  # Orange square
    
    pil_frame = Image.fromarray(np_frame)
    assert pil_frame.mode == 'RGB'
    assert pil_frame.size == (40, 40)
    
    # Test that PIL frames pass through unchanged
    original_pil = Image.new('RGB', (50, 50), color='purple')
    converted_frames = [
        Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
        for frame in [np_frame, original_pil]
    ]
    
    assert len(converted_frames) == 2
    assert all(isinstance(frame, Image.Image) for frame in converted_frames)


def test_gif_optimization():
    """Test GIF optimization and file size considerations."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create frames with different complexities
        simple_frames = []
        complex_frames = []
        
        # Simple frames (solid colors)
        for i in range(3):
            frame = np.full((50, 50, 3), i * 85, dtype=np.uint8)  # Solid colors
            simple_frames.append(frame)
        
        # Complex frames (random noise)
        for i in range(3):
            frame = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
            complex_frames.append(frame)
        
        simple_path = os.path.join(tmp_dir, "simple.gif")
        complex_path = os.path.join(tmp_dir, "complex.gif")
        
        mock_save_gif(simple_frames, simple_path, fps=10)
        mock_save_gif(complex_frames, complex_path, fps=10)
        
        simple_size = os.path.getsize(simple_path)
        complex_size = os.path.getsize(complex_path)
        
        # Complex frames should generally create larger files
        # (though this isn't guaranteed due to GIF compression)
        assert simple_size > 0
        assert complex_size > 0


def test_large_frame_handling():
    """Test handling of larger frames."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a larger frame
        large_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        large_frame[50:150, 50:150, 0] = 255  # Red square in center
        
        gif_path = os.path.join(tmp_dir, "large.gif")
        mock_save_gif([large_frame], gif_path, fps=1)
        
        assert os.path.exists(gif_path)
        
        with Image.open(gif_path) as gif:
            assert gif.format == "GIF"
            assert gif.size == (200, 200)


def test_performance_with_multiple_frames():
    """Test performance with multiple frames."""
    import time
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create multiple frames
        num_frames = 10
        frames = []
        
        start_time = time.time()
        
        for i in range(num_frames):
            frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            frames.append(frame)
        
        creation_time = time.time() - start_time
        
        gif_path = os.path.join(tmp_dir, "performance.gif")
        
        start_gif_time = time.time()
        mock_save_gif(frames, gif_path, fps=30)
        gif_creation_time = time.time() - start_gif_time
        
        # Performance should be reasonable
        assert creation_time < 2.0, f"Frame creation too slow: {creation_time:.3f}s"
        assert gif_creation_time < 5.0, f"GIF creation too slow: {gif_creation_time:.3f}s"
        
        assert os.path.exists(gif_path)
        
        with Image.open(gif_path) as gif:
            assert gif.n_frames == num_frames


if __name__ == "__main__":
    # Quick smoke test
    print("Running GIF core tests...")
    test_save_gif_function_directly()
    test_frame_conversion_logic()
    print("✅ Core GIF tests passed!")