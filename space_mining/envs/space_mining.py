from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .renderer import Renderer


class SpaceMining(gym.Env):
    """
    Space Mining Environment (Simplified Single-Agent Version)

    Agent (mining robot) must collect resources from asteroids
    and return them to the mothership while managing energy and avoiding obstacles.

    Features:
    - Physics simulation with gravity, inertia, and collisions
    - Dynamic resource distribution
    - Energy management
    - Partial observability
    This environment follows the Gymnasium API for compatibility.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        max_episode_steps: int = 1200,
        grid_size: int = 80,
        max_asteroids: int = 12,
        max_resource_per_asteroid: int = 40,
        observation_radius: int = 15,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.grid_size = grid_size
        self.max_asteroids = max(max_asteroids, 18)
        self.max_resource_per_asteroid = max_resource_per_asteroid
        self.observation_radius = observation_radius
        self.render_mode = render_mode

        self.dt: float = 0.1
        self.mass: float = 3.0
        self.max_force: float = 20.0
        self.drag_coef: float = 0.02
        self.gravity_strength: float = 0.01
        self.obstacle_penalty: float = -10.0
        self.energy_consumption_rate: float = 0.05
        self.mining_energy_cost: float = 1.0

        self.steps_count: int = 0
        self.mothership_pos: np.ndarray = np.array([grid_size / 2, grid_size / 2])

        self.action_space: spaces.Box = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        self.max_obs_asteroids: int = 15
        self.max_inventory: int = 100
        self.mining_range: float = 8.0

        agent_state_dim: int = 6
        asteroids_dim: int = self.max_obs_asteroids * 3
        mothership_dim: int = 2

        self.observation_space: spaces.Box = spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(agent_state_dim + asteroids_dim + mothership_dim,),
            dtype=np.float32,
        )

        self.renderer: Renderer = Renderer(self)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.steps_count = 0
        self.collision_count = 0

        self.agent_position = self.np_random.uniform(low=0, high=self.grid_size, size=(2,))
        self.agent_velocity = np.zeros(2, dtype=np.float32)
        self.agent_energy = 150.0
        self.agent_inventory = 0.0
        self.cumulative_mining_amount = 0.0
        self.discovered_asteroids = set()

        min_asteroids = min(8, self.max_asteroids)
        low = min_asteroids
        high = min(12, self.max_asteroids) + 1
        if low >= high:
            low = max(6, self.max_asteroids)
        num_asteroids = self.np_random.integers(low, high)
        self.asteroid_positions = self.np_random.uniform(
            low=15, high=self.grid_size - 15, size=(num_asteroids, 2)
        )
        self.asteroid_resources = self.np_random.uniform(low=25, high=40, size=(num_asteroids,))

        num_obstacles = self.np_random.integers(4, 8)
        self.obstacle_positions = self.np_random.uniform(
            low=20, high=self.grid_size - 20, size=(num_obstacles, 2)
        )
        self.obstacle_velocities = self.np_random.uniform(
            low=-0.2, high=0.2, size=(num_obstacles, 2)
        )

        self.prev_min_asteroid_distance = float("inf")
        self.prev_inventory = 0.0
        self.prev_energy = self.agent_energy
        self.prev_distance_to_mothership = np.linalg.norm(self.agent_position - self.mothership_pos)

        observation = self._get_observation()
        info = {"total_resources_collected": 0, "obstacle_collisions": 0, "energy_remaining": self.agent_energy}
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.steps_count += 1

        required_attrs = ("agent_position", "agent_velocity", "asteroid_positions", "asteroid_resources")
        if not all(hasattr(self, attr) for attr in required_attrs):
            preset_energy = getattr(self, "agent_energy", None)
            _, _ = self.reset()
            if preset_energy is not None:
                self.agent_energy = float(preset_energy)
                self.prev_energy = self.agent_energy

        if hasattr(self, "tried_depleted_asteroid"):
            delattr(self, "tried_depleted_asteroid")

        thrust = action[:2] * self.max_force
        mine = action[2] > 0.5

        reward = 0.0

        if self.agent_energy <= 0:
            terminated = True
            truncated = False
            reward = -1.0
            observation = self._get_observation()
            info = self._get_info()
            return observation, reward, terminated, truncated, info

        acceleration = thrust / self.mass
        drag_acceleration = -self.drag_coef * self.agent_velocity / self.mass
        to_mothership = self.mothership_pos - self.agent_position
        distance_to_mothership = np.linalg.norm(to_mothership)
        if distance_to_mothership > 0:
            gravity_acceleration = (to_mothership / distance_to_mothership) * self.gravity_strength
        else:
            gravity_acceleration = np.zeros(2)

        self.agent_velocity += (acceleration + drag_acceleration + gravity_acceleration) * self.dt
        self.agent_position += self.agent_velocity * self.dt

        boundary_margin = 5.0
        for axis in range(2):
            if self.agent_position[axis] < boundary_margin:
                self.agent_position[axis] = boundary_margin
                self.agent_velocity[axis] = -0.3 * self.agent_velocity[axis]
                reward += -1.0
            elif self.agent_position[axis] > self.grid_size - boundary_margin:
                self.agent_position[axis] = self.grid_size - boundary_margin
                self.agent_velocity[axis] = -0.3 * self.agent_velocity[axis]
                reward += -1.0

        energy_used = self.energy_consumption_rate * 0.5
        energy_used += np.sum(np.abs(thrust)) * 0.01
        if mine:
            energy_used += self.mining_energy_cost * 0.5
        self.agent_energy -= energy_used
        if not hasattr(self, "last_energy_used"):
            self.last_energy_used = 0.0
        self.last_energy_used = energy_used

        if self.agent_energy <= 0:
            self.agent_energy = 0
            reward += -10.0
            terminated = True
        else:
            terminated = False

        obstacle_collisions = 0
        for obstacle_pos in self.obstacle_positions:
            distance = np.linalg.norm(self.agent_position - obstacle_pos)
            if distance < 1.5:
                reward += -10.0
                obstacle_collisions += 1
                self.collision_count += 1
                if not hasattr(self, "last_collision_step"):
                    self.last_collision_step = 0
                self.last_collision_step = self.steps_count
                to_obstacle = self.agent_position - obstacle_pos
                if np.linalg.norm(to_obstacle) > 0:
                    to_obstacle = to_obstacle / np.linalg.norm(to_obstacle)
                    self.agent_velocity += to_obstacle * 2.0
        if self.collision_count >= 18:
            print(f"[EPISODE END] Step {self.steps_count}: Too many collisions, terminating episode.")
            terminated = True

        if mine and self.agent_energy > 0 and self.agent_inventory < self.max_inventory:
            mined_something = False
            tried_depleted_asteroid = False
            for i, asteroid_pos in enumerate(self.asteroid_positions):
                distance = np.linalg.norm(self.agent_position - asteroid_pos)
                if distance < self.mining_range:
                    if self.asteroid_resources[i] < 0.1:
                        tried_depleted_asteroid = True
                        continue
                    mining_efficiency = 0.6
                    max_possible = min(
                        self.asteroid_resources[i] * mining_efficiency,
                        self.max_inventory - self.agent_inventory,
                    )
                    if max_possible > 0:
                        self.asteroid_resources[i] -= max_possible
                        if self.asteroid_resources[i] < 0.1:
                            self.asteroid_resources[i] = 0.0
                        self.agent_inventory += max_possible
                        if not hasattr(self, "cumulative_mining_amount"):
                            self.cumulative_mining_amount = 0.0
                        self.cumulative_mining_amount += max_possible
                        if not hasattr(self, "last_mining_info"):
                            self.last_mining_info = {}
                        self.last_mining_info = {
                            "step": self.steps_count,
                            "asteroid_id": i,
                            "extracted": max_possible,
                            "inventory": self.agent_inventory,
                            "cumulative_mining": self.cumulative_mining_amount,
                            "asteroid_depleted": self.asteroid_resources[i] <= 0,
                        }
                        self.mining_asteroid_id = i
                        reward += max_possible * 8.0
                        mined_something = True
                        self.agent_velocity *= 0.8
                        break
            if not mined_something:
                if tried_depleted_asteroid:
                    reward -= 0.2
                    self.tried_depleted_asteroid = True
                else:
                    reward -= 0.1
                if hasattr(self, "mining_asteroid_id"):
                    delattr(self, "mining_asteroid_id")
        elif mine and self.agent_inventory >= self.max_inventory:
            reward -= 0.2
            if hasattr(self, "mining_asteroid_id"):
                delattr(self, "mining_asteroid_id")
        elif not mine:
            if hasattr(self, "mining_asteroid_id"):
                delattr(self, "mining_asteroid_id")

        distance_to_mothership = np.linalg.norm(self.agent_position - self.mothership_pos)
        if distance_to_mothership < 12.0 and self.agent_inventory > 0:
            delivered_amount = self.agent_inventory
            reward += delivered_amount * 12.0
            if not hasattr(self, "last_delivery_info"):
                self.last_delivery_info = {}
            energy_recharged = 150.0 - self.agent_energy
            self.last_delivery_info = {
                "step": self.steps_count,
                "delivered": delivered_amount,
                "energy_recharged": energy_recharged,
            }
            self.agent_inventory = 0
            self.agent_energy = 150.0
            reward += energy_recharged * 0.5

        for i in range(len(self.obstacle_positions)):
            self.obstacle_positions[i] += self.obstacle_velocities[i] * self.dt
            for axis in range(2):
                if self.obstacle_positions[i][axis] < 0 or self.obstacle_positions[i][axis] > self.grid_size:
                    self.obstacle_velocities[i][axis] *= -1

        observation = self._get_observation()

        truncated = self.steps_count >= self.max_episode_steps
        if truncated:
            print(
                (
                    f"[EPISODE END] Step {self.steps_count}: Time limit reached "
                    f"({self.max_episode_steps} steps) - Episode truncated"
                )
            )

        if np.all(self.asteroid_resources < 0.1):
            terminated = True
            info = self._get_info()
            info["exploration_complete"] = True
            print((f"[EPISODE END] Step {self.steps_count}: All asteroids depleted " f"- Episode completed successfully"))

        info = self._get_info()
        info["obstacle_collisions"] = obstacle_collisions

        advanced_reward, reward_info = self.compute_reward(action, observation, info)
        reward += advanced_reward

        fitness_score = self.compute_fitness_score()
        info["fitness_score"] = fitness_score
        info.update(reward_info)
        info["immediate_rewards"] = {"mining_reward": reward - advanced_reward, "total_reward": reward}

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        agent_state = np.concatenate([self.agent_position, self.agent_velocity, [self.agent_energy], [self.agent_inventory]])
        asteroid_obs = np.zeros((self.max_obs_asteroids, 3), dtype=np.float32)
        asteroid_count = 0
        for i, asteroid_pos in enumerate(self.asteroid_positions):
            if self.asteroid_resources[i] < 0.1:
                continue
            rel_pos = asteroid_pos - self.agent_position
            distance = np.linalg.norm(rel_pos)
            if distance <= self.observation_radius and asteroid_count < self.max_obs_asteroids:
                asteroid_obs[asteroid_count] = np.concatenate([rel_pos, [self.asteroid_resources[i]]])
                asteroid_count += 1
        mothership_rel_pos = self.mothership_pos - self.agent_position
        observation = np.concatenate([agent_state, asteroid_obs.flatten(), mothership_rel_pos])
        return observation.astype(self.observation_space.dtype, copy=False)

    def _get_info(self) -> Dict[str, Any]:
        info = {
            "agent_position": self.agent_position.copy(),
            "agent_velocity": self.agent_velocity.copy(),
            "agent_energy": float(self.agent_energy),
            "agent_inventory": float(self.agent_inventory),
            "asteroid_resources": self.asteroid_resources.copy(),
            "mothership_pos": self.mothership_pos.copy(),
            "asteroid_positions": self.asteroid_positions.copy(),
            "collision_count": self.collision_count,
            "steps_count": self.steps_count,
        }
        if hasattr(self, "cumulative_mining_amount"):
            info["cumulative_mining_amount"] = float(self.cumulative_mining_amount)
        else:
            info["cumulative_mining_amount"] = 0.0
        info["mining_asteroid_id"] = getattr(self, "mining_asteroid_id", None)
        return info

    def compute_fitness_score(self) -> float:
        resources_collected = self.agent_inventory
        energy_remaining = self.agent_energy / 100.0
        remaining_resources = np.sum(self.asteroid_resources)
        total_resources_initial = self.max_resource_per_asteroid * self.max_asteroids
        resource_depletion_ratio = 1.0 - (remaining_resources / total_resources_initial)

        nearest_asteroid_dist = float("inf")
        for i, asteroid_pos in enumerate(self.asteroid_positions):
            if self.asteroid_resources[i] >= 0.1:
                dist = np.linalg.norm(self.agent_position - asteroid_pos)
                nearest_asteroid_dist = min(nearest_asteroid_dist, dist)
        if nearest_asteroid_dist == float("inf"):
            nearest_asteroid_dist = 0
        else:
            nearest_asteroid_dist = 1.0 - min(1.0, nearest_asteroid_dist / self.grid_size)

        distance_to_mothership = np.linalg.norm(self.agent_position - self.mothership_pos)
        mothership_proximity = 1.0 - min(1.0, distance_to_mothership / self.grid_size)

        fitness = (
            resources_collected * 50.0
            + energy_remaining * 300.0
            + resource_depletion_ratio * 200.0
            + nearest_asteroid_dist * 100.0 * (1 - resource_depletion_ratio)
            + mothership_proximity * 100.0 * (self.agent_inventory > 0)
        )
        completion_bonus = resource_depletion_ratio * 500.0
        efficiency_bonus = (self.steps_count > 0) * (resource_depletion_ratio / max(1, self.steps_count)) * 1000.0
        survival_bonus = self.steps_count * 0.5
        fitness += completion_bonus + efficiency_bonus + survival_bonus
        return fitness

    def render(self) -> Optional[np.ndarray]:
        return self.renderer.render()

    def close(self) -> None:
        self.renderer.close()

    def compute_reward(self, action: np.ndarray, observation: np.ndarray, info: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        SPEED_LIMIT = 10.0
        EFFICIENCY_THRESHOLD = 0.5
        EXPLORATION_BONUS = 3.0
        PATH_EFFICIENCY_BONUS = 2.0
        MINING_GUIDANCE_BONUS = 2.0
        DELIVERY_GUIDANCE_BONUS = 3.0

        speed_penalty = 0.0
        efficiency_reward = 0.0
        exploration_reward = 0.0
        path_efficiency_reward = 0.0
        mining_guidance_reward = 0.0
        delivery_guidance_reward = 0.0

        speed = np.linalg.norm(observation[2:4])
        if speed > SPEED_LIMIT:
            speed_penalty = -0.05 * (speed - SPEED_LIMIT) ** 2

        energy = observation[4]
        energy_ratio = energy / 150.0
        if energy_ratio > EFFICIENCY_THRESHOLD:
            efficiency_reward = 1.0 * energy_ratio

        if not hasattr(self, "discovered_asteroids"):
            self.discovered_asteroids = set()
        for i, asteroid_pos in enumerate(self.asteroid_positions):
            if self.asteroid_resources[i] <= 0.1:
                continue
            distance = np.linalg.norm(self.agent_position - asteroid_pos)
            if distance <= self.observation_radius and i not in self.discovered_asteroids:
                self.discovered_asteroids.add(i)
                exploration_reward += EXPLORATION_BONUS

        inventory = observation[5]
        if inventory > 0:
            distance_to_mothership = np.linalg.norm(observation[0:2] - self.mothership_pos)
            if distance_to_mothership < 15.0:
                path_efficiency_reward = PATH_EFFICIENCY_BONUS * 2.0 * (1.0 - distance_to_mothership / 15.0)
                delivery_guidance_reward = DELIVERY_GUIDANCE_BONUS * (1.0 - distance_to_mothership / 15.0)
        else:
            nearest_asteroid_dist = float("inf")
            for i, asteroid_pos in enumerate(self.asteroid_positions):
                if self.asteroid_resources[i] > 0.1:
                    dist = np.linalg.norm(self.agent_position - asteroid_pos)
                    nearest_asteroid_dist = min(nearest_asteroid_dist, dist)
            if nearest_asteroid_dist < float("inf"):
                if nearest_asteroid_dist < 10.0:
                    path_efficiency_reward = PATH_EFFICIENCY_BONUS * 2.0 * (1.0 - nearest_asteroid_dist / 10.0)
                    mining_guidance_reward = MINING_GUIDANCE_BONUS * (1.0 - nearest_asteroid_dist / 10.0)

        if energy_ratio < 0.3 and inventory == 0:
            distance_to_mothership = np.linalg.norm(observation[0:2] - self.mothership_pos)
            if distance_to_mothership < 20.0:
                delivery_guidance_reward += 1.0 * (1.0 - distance_to_mothership / 20.0)

        total_reward = (
            speed_penalty + efficiency_reward + exploration_reward + path_efficiency_reward + mining_guidance_reward + delivery_guidance_reward
        )
        return total_reward, {
            "speed_penalty": speed_penalty,
            "efficiency_reward": efficiency_reward,
            "exploration_reward": exploration_reward,
            "path_efficiency_reward": path_efficiency_reward,
            "mining_guidance_reward": mining_guidance_reward,
            "delivery_guidance_reward": delivery_guidance_reward,
        }

__all__ = ["SpaceMining"]