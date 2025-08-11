import numpy as np

from space_mining.envs import make_env


def test_render_rgb_array_shape():
    env = make_env(render_mode="rgb_array")
    env.reset()
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (800, 800, 3)
    env.close()