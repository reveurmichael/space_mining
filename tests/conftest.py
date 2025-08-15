"""Shared pytest fixtures for space_mining tests."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture(scope="session")
def shared_test_model():
    """Create a mock shared trained model for tests that need one.
    
    Since ML dependencies are not available, this returns a mock path.
    """
    # Return a mock model path for tests that expect it
    return "/tmp/mock_model.zip"


@pytest.fixture
def minimal_frames():
    """Create minimal test frames for GIF testing."""
    import numpy as np
    
    frames = []
    for i in range(3):  # Just 3 frames
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        frame[:, :, i % 3] = 128  # Different colors but not too bright
        frames.append(frame)
    
    return frames


@pytest.fixture
def tmp_gif_path(tmp_path):
    """Create a temporary GIF path for testing."""
    return tmp_path / "test.gif"