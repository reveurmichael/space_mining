"""Quickstart: run SpaceMining with a random policy.

Overview
- Creates the environment in human render mode with a 1200-step cap
- Samples random actions for 500 steps, resetting on episode end
- Prints total reward per episode for quick feedback

Intended use
- Sanity-check installation and rendering
- Provide a minimal baseline run for integration tests
"""

from space_mining import make_env


def main() -> None:
    """Run a short random-action session in SpaceMining.

    Steps
    1. Construct the environment with human rendering.
    2. Reset to obtain the initial observation.
    3. Loop for a fixed number of steps, sampling random actions.
    4. Accumulate reward and reset the episode on termination/truncation.
    5. Close the environment cleanly at the end.

    Notes
    - This function has no side effects beyond console output.
    - Use this script to verify graphics and environment installation quickly.
    """

    # Create the environment (human-mode shows a real-time window)
    env = make_env(render_mode="human", max_episode_steps=1200)

    # Initialize episode state
    obs, info = env.reset()
    total_reward = 0.0

    # Roll out a short random-control session
    for _ in range(500):
        # Sample a random action from the environment's action space
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # If the episode ends early, report and start a new one
        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward}")
            obs, info = env.reset()
            total_reward = 0.0

    # Always close the environment to release resources
    env.close()


if __name__ == "__main__":
    main()
