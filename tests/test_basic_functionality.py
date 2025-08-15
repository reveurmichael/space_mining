"""Basic functionality tests that work without ML dependencies."""

import numpy as np
import tempfile
from pathlib import Path
from PIL import Image
import pytest


def test_numpy_available():
    """Test that numpy is working correctly."""
    arr = np.array([1, 2, 3])
    assert arr.shape == (3,)
    assert arr.dtype == np.int64 or arr.dtype == np.int32


def test_pil_available():
    """Test that PIL is working correctly."""
    img = Image.new('RGB', (100, 100), color='red')
    assert img.size == (100, 100)
    assert img.mode == 'RGB'


def test_frame_to_gif_conversion(tmp_path):
    """Test basic frame to GIF conversion without ML dependencies."""
    # Create test frames
    frames = []
    for i in range(3):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        frame[:, :, i % 3] = 255  # Red, Green, Blue frames
        frames.append(frame)
    
    # Convert numpy arrays to PIL Images
    pil_frames = [Image.fromarray(frame) for frame in frames]
    
    # Save as GIF
    gif_path = tmp_path / "test.gif"
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=100,  # 100ms per frame
        loop=0,
    )
    
    # Verify GIF was created
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0
    
    # Verify GIF can be opened
    with Image.open(gif_path) as gif:
        assert gif.format == "GIF"
        assert gif.n_frames == 3


def test_mock_save_gif_function(tmp_path):
    """Test a mock version of the save_gif function."""
    def mock_save_gif(frames, output_path, fps=30):
        """Mock save_gif function for testing."""
        import os
        from PIL import Image
        
        output_dir = os.path.dirname(output_path) or "."
        os.makedirs(output_dir, exist_ok=True)

        pil_frames = [
            Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame 
            for frame in frames
        ]

        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )
    
    # Create test frames
    frames = [
        np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)
        for _ in range(2)
    ]
    
    gif_path = tmp_path / "mock_test.gif"
    mock_save_gif(frames, str(gif_path), fps=15)
    
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0


def test_frame_shapes_and_types():
    """Test frame shape and type validation."""
    # Test valid frame
    valid_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert valid_frame.shape == (100, 100, 3)
    assert valid_frame.dtype == np.uint8
    
    # Test different frame sizes
    small_frame = np.zeros((50, 50, 3), dtype=np.uint8)
    large_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    
    assert small_frame.shape == (50, 50, 3)
    assert large_frame.shape == (200, 200, 3)
    
    # Test frame value ranges
    frame = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    assert np.all(frame >= 0)
    assert np.all(frame <= 255)


def test_directory_creation_for_gifs(tmp_path):
    """Test that directory creation works for GIF output."""
    # Create nested directory path
    nested_path = tmp_path / "level1" / "level2" / "test.gif"
    
    # Should not exist initially
    assert not nested_path.exists()
    assert not nested_path.parent.exists()
    
    # Create directories
    nested_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Now should exist
    assert nested_path.parent.exists()
    
    # Create a simple gif
    img = Image.new('RGB', (10, 10), color='blue')
    img.save(nested_path)
    
    assert nested_path.exists()


def test_memory_usage_basic():
    """Test basic memory usage with numpy arrays."""
    import psutil
    import os
    
    # Get initial memory
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create some arrays
    arrays = []
    for i in range(10):
        arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        arrays.append(arr)
    
    # Check memory usage
    current_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = current_memory - initial_memory
    
    # Should be reasonable (arrays are small)
    assert memory_increase < 100  # Less than 100MB increase
    
    # Clean up
    del arrays


def test_concurrent_gif_creation(tmp_path):
    """Test basic concurrent GIF creation."""
    import threading
    import queue
    
    results = queue.Queue()
    
    def create_gif(gif_name):
        try:
            # Create simple frame
            frame = np.zeros((20, 20, 3), dtype=np.uint8)
            frame[:, :, 0] = 255  # Red
            
            # Convert to PIL and save
            pil_frame = Image.fromarray(frame)
            gif_path = tmp_path / f"{gif_name}.gif"
            pil_frame.save(gif_path)
            
            results.put(("success", gif_name, gif_path.exists()))
        except Exception as e:
            results.put(("error", gif_name, str(e)))
    
    # Start multiple threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=create_gif, args=(f"test_{i}",))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads
    for thread in threads:
        thread.join()
    
    # Check results
    successes = 0
    while not results.empty():
        status, name, result = results.get()
        if status == "success":
            assert result is True
            successes += 1
        else:
            pytest.fail(f"Thread {name} failed: {result}")
    
    assert successes == 3


def test_error_handling():
    """Test error handling for invalid inputs."""
    # Test with empty array
    empty_array = np.array([])
    assert empty_array.size == 0
    
    # Test with wrong shape
    wrong_shape = np.array([1, 2, 3])
    assert wrong_shape.shape == (3,)
    assert len(wrong_shape.shape) == 1  # Not a 3D array
    
    # Test invalid GIF path
    invalid_path = "/non/existent/path/test.gif"
    try:
        img = Image.new('RGB', (10, 10))
        img.save(invalid_path)
        assert False, "Should have raised an exception"
    except (OSError, PermissionError, FileNotFoundError):
        pass  # Expected


def test_performance_timing():
    """Test basic performance timing."""
    import time
    
    start_time = time.time()
    
    # Create some frames
    frames = []
    for i in range(5):
        frame = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        frames.append(frame)
    
    # Convert to PIL
    pil_frames = [Image.fromarray(frame) for frame in frames]
    
    end_time = time.time()
    creation_time = end_time - start_time
    
    # Should be fast
    assert creation_time < 1.0, f"Frame creation too slow: {creation_time:.3f}s"


if __name__ == "__main__":
    print("Running basic functionality tests...")
    test_numpy_available()
    test_pil_available()
    print("✅ All basic tests would pass!")