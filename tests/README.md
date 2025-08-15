# Space Mining Test Suite

This directory contains comprehensive tests for the Space Mining environment, with a focus on renderer functionality and GIF generation capabilities.

## Test Files Overview

### Core Environment Tests
- `test_env_creation.py` - Environment creation and basic setup
- `test_env_dynamics.py` - Environment physics and dynamics
- `test_env_spaces.py` - Action and observation spaces
- `test_env_step.py` - Basic environment stepping
- `test_env_render.py` - Basic rendering functionality

### Agent Tests
- `test_ppo_agent_learn.py` - PPO agent training
- `test_ppo_agent_load.py` - Model loading and saving
- `test_ppo_agent_predict.py` - Agent prediction functionality

### New Visualization Tests
- **`test_renderer.py`** - Comprehensive renderer functionality tests
- **`test_gif_generation.py`** - GIF creation and trajectory generation tests
- **`test_integration.py`** - End-to-end workflow tests

### Test Infrastructure
- `conftest.py` - Shared fixtures for performance optimization

## New Test Features

### Renderer Tests (`test_renderer.py`)
Tests all visualization-related functionality that was moved from `space_mining.py` to `renderer.py`:

- **Cosmic Background**: Starfield, nebulae, galaxies, space dust, auroras
- **Visual Effects Methods**: All moved visualization methods are tested
- **Zoom System**: Dynamic zoom functionality 
- **Animation Systems**: Event timeline, score popups, delivery particles
- **Combo System**: Mining combo detection and display
- **Game Over Screen**: Statistics and final screen rendering
- **Performance**: Render timing optimization
- **Integration**: Renderer working with trained agents

### GIF Generation Tests (`test_gif_generation.py`)
Comprehensive testing of GIF creation functionality:

- **Frame Types**: NumPy arrays, PIL Images, mixed formats
- **Trajectory Generation**: From existing trained models
- **Complete Workflow**: Model → Frames → GIF
- **Parameter Testing**: Different FPS, deterministic vs stochastic
- **Error Handling**: Invalid inputs and edge cases
- **Performance**: GIF creation timing
- **API Testing**: Both direct and wrapper APIs

### Integration Tests (`test_integration.py`)
Real-world usage scenarios:

- **Model-to-GIF Workflow**: Complete pipeline testing
- **Visual Effects Integration**: Effects working during model execution
- **Quality Optimization**: Different GIF parameters
- **Memory Usage**: Resource consumption monitoring
- **Concurrent Generation**: Multiple GIF creation
- **Error Recovery**: Graceful error handling

## Performance Optimizations

### Shared Fixtures
- **`shared_test_model`**: Session-scoped trained model (created once per test session)
- **`fast_env`**: Short episode environments for speed
- **`minimal_frames`**: Pre-created test frames

### Speed Optimizations
- Minimal training timesteps (10-20 steps)
- Short episodes (5-50 steps) 
- Small frame counts (2-10 frames)
- Reuse of trained models across tests
- Parallel test execution support

## Running the Tests

### All Tests
```bash
pytest tests/ -v
```

### Specific Test Categories
```bash
# Renderer tests only
pytest tests/test_renderer.py -v

# GIF generation tests only  
pytest tests/test_gif_generation.py -v

# Integration tests only
pytest tests/test_integration.py -v

# All new visualization tests
pytest tests/test_renderer.py tests/test_gif_generation.py tests/test_integration.py -v
```

### Performance Testing
```bash
# Run with timing
pytest tests/ -v --durations=10

# Run specific performance-sensitive tests
pytest tests/test_renderer.py::test_renderer_performance -v
pytest tests/test_gif_generation.py::test_gif_performance -v
```

## Test Design Principles

### Fast Execution
- All tests designed to complete quickly (< 30 seconds total for new tests)
- Minimal model training using existing patterns
- Short trajectories and small frame counts
- Efficient resource cleanup

### Comprehensive Coverage
- Tests cover all moved renderer functionality
- Both success and error cases tested
- Real-world usage patterns simulated
- Performance characteristics verified

### Reusable Models
- Shared test models to avoid repeated training
- Session-scoped fixtures for efficiency
- Consistent test data across test runs

## Expected Test Times

- **Renderer Tests**: ~15-20 seconds
- **GIF Generation Tests**: ~10-15 seconds  
- **Integration Tests**: ~20-25 seconds
- **Total New Tests**: ~45-60 seconds

The tests are designed to be fast enough for frequent execution during development while providing comprehensive coverage of all visualization functionality.