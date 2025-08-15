from typing import Any, Dict, Optional, Tuple
import math

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

        # Animation data structures
        self.delivery_particles = []
        self.agent_trail = []
        self.score_popups = []
        self.collision_flash_timer = 0.0
        self.screen_shake_timer = 0.0
        self.mining_beam_offset = 0.0
        
        # Enhanced cosmic background system
        self.starfield_layers = []
        self.nebula_clouds = []
        self.distant_galaxies = []
        self.space_dust = []
        self.cosmic_time = 0.0
        self._initialize_cosmic_background()
        
        # Zoom system
        self.zoom_level = 1.0
        self.target_zoom = 1.0
        self.zoom_speed = 0.02
        
        # Game over screen state
        self.game_over_state = {
            "active": False,
            "fade_alpha": 0,
            "final_stats": {},
            "success": False
        }
        
        # Floating event timeline
        self.event_timeline = []  # List of recent events with timestamps
        self.max_timeline_events = 5
        
        # Score combo system
        self.combo_state = {
            "chain_count": 0,
            "last_mining_step": 0,
            "combo_window": 50,  # Steps to maintain combo
            "display_timer": 0,
            "combo_alpha": 0
        }

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

        # Reset animation data structures
        self.delivery_particles = []
        self.agent_trail = []
        self.score_popups = []
        self.collision_flash_timer = 0.0
        self.screen_shake_timer = 0.0
        self.mining_beam_offset = 0.0

        # Reset cosmic background and game over state
        self._initialize_cosmic_background()
        self.game_over_state = {
            "active": False,
            "fade_alpha": 0,
            "final_stats": {},
            "success": False
        }

        # Reset event timeline and combo system
        self.event_timeline = []
        self.combo_state = {
            "chain_count": 0,
            "last_mining_step": 0,
            "combo_window": 50,
            "display_timer": 0,
            "combo_alpha": 0
        }

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
            self._trigger_game_over(success=False)
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
                # Trigger collision effects
                self.collision_flash_timer = 0.3  # Flash for 0.3 seconds
                self.screen_shake_timer = 0.2  # Shake for 0.2 seconds
                # Add to event timeline
                self._add_timeline_event("collision", "Collision!", (255, 100, 100))
                to_obstacle = self.agent_position - obstacle_pos
                if np.linalg.norm(to_obstacle) > 0:
                    to_obstacle = to_obstacle / np.linalg.norm(to_obstacle)
                    self.agent_velocity += to_obstacle * 2.0
        if self.collision_count >= 18:
            print(f"[EPISODE END] Step {self.steps_count}: Too many collisions, terminating episode.")
            terminated = True
            self._trigger_game_over(success=False)

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
                        # Add score popup for mining
                        self._add_score_popup(f"+{max_possible:.1f}", asteroid_pos.copy(), (255, 255, 0))
                        # Add to event timeline
                        self._add_timeline_event("mining", f"+{max_possible:.1f}", (255, 255, 0))
                        # Update combo system
                        self._process_mining_combo()
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
            # Spawn delivery particles
            self._spawn_delivery_particles(self.agent_position.copy(), self.mothership_pos.copy())
            # Add score popup for delivery
            self._add_score_popup(f"+{delivered_amount:.1f}", self.agent_position.copy(), (0, 255, 0))
            # Add to event timeline
            self._add_timeline_event("delivery", f"Delivered +{delivered_amount:.1f}", (0, 255, 0))
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

        # Update animations and cosmic background
        self._update_animations()
        self._update_cosmic_background()
        self._update_zoom()

        observation = self._get_observation()

        truncated = self.steps_count >= self.max_episode_steps
        if truncated:
            self._trigger_game_over(success=False)
            print(
                (
                    f"[EPISODE END] Step {self.steps_count}: Time limit reached "
                    f"({self.max_episode_steps} steps) - Episode truncated"
                )
            )

        if np.all(self.asteroid_resources < 0.1):
            terminated = True
            self._trigger_game_over(success=True)
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

    def _update_animations(self) -> None:
        """Update all animation states."""
        # Update agent trail
        self.agent_trail.append({"pos": self.agent_position.copy(), "alpha": 255})
        # Fade existing trail points
        for trail_point in self.agent_trail:
            trail_point["alpha"] -= 15
        # Remove faded trail points
        self.agent_trail = [p for p in self.agent_trail if p["alpha"] > 0]

        # Update delivery particles
        for particle in self.delivery_particles:
            particle["progress"] += 0.05
            if particle["progress"] >= 1.0:
                particle["progress"] = 1.0
        # Remove completed particles
        self.delivery_particles = [p for p in self.delivery_particles if p["progress"] < 1.0]

        # Update score popups
        for popup in self.score_popups:
            popup["pos"][1] -= 0.3  # Move upward
            popup["alpha"] -= 5
        # Remove faded popups
        self.score_popups = [p for p in self.score_popups if p["alpha"] > 0]

        # Update collision effects
        if self.collision_flash_timer > 0:
            self.collision_flash_timer -= self.dt
        if self.screen_shake_timer > 0:
            self.screen_shake_timer -= self.dt

        # Update mining beam animation
        self.mining_beam_offset += 0.2

        # Update event timeline
        self._update_event_timeline()

        # Update combo system
        self._update_combo_system()

    def _spawn_delivery_particles(self, start_pos: np.ndarray, target_pos: np.ndarray) -> None:
        """Spawn glowing particles for resource delivery animation."""
        for _ in range(10):
            particle = {
                "start_pos": start_pos.copy(),
                "target_pos": target_pos.copy(),
                "progress": 0.0
            }
            self.delivery_particles.append(particle)

    def _add_score_popup(self, text: str, pos: np.ndarray, color: tuple) -> None:
        """Add a floating score popup."""
        popup = {
            "text": text,
            "pos": pos.copy(),
            "alpha": 255,
            "color": color
        }
        self.score_popups.append(popup)

    def _initialize_cosmic_background(self) -> None:
        """Initialize enhanced cosmic background with nebulae, galaxies, and stars."""
        # Reset cosmic time
        self.cosmic_time = 0.0
        
        # Create enhanced starfield layers
        self.starfield_layers = []
        
        # Layer 1: Distant stars (slow, small, dim)
        layer1_stars = []
        for _ in range(150):  # More stars for cosmic feel
            star = {
                "x": np.random.uniform(0, 1200),
                "y": np.random.uniform(0, 900),
                "size": 1,
                "brightness": np.random.randint(40, 120),
                "speed": 0.15,
                "twinkle_offset": np.random.uniform(0, 2 * math.pi),
                "color_type": np.random.choice(["white", "blue", "yellow", "red"])
            }
            layer1_stars.append(star)
        self.starfield_layers.append(layer1_stars)
        
        # Layer 2: Medium stars
        layer2_stars = []
        for _ in range(80):
            star = {
                "x": np.random.uniform(0, 1200),
                "y": np.random.uniform(0, 900),
                "size": 2,
                "brightness": np.random.randint(80, 180),
                "speed": 0.4,
                "twinkle_offset": np.random.uniform(0, 2 * math.pi),
                "color_type": np.random.choice(["white", "blue", "yellow"])
            }
            layer2_stars.append(star)
        self.starfield_layers.append(layer2_stars)
        
        # Layer 3: Bright foreground stars
        layer3_stars = []
        for _ in range(40):
            star = {
                "x": np.random.uniform(0, 1200),
                "y": np.random.uniform(0, 900),
                "size": 3,
                "brightness": np.random.randint(120, 255),
                "speed": 0.8,
                "twinkle_offset": np.random.uniform(0, 2 * math.pi),
                "color_type": np.random.choice(["white", "blue", "yellow"])
            }
            layer3_stars.append(star)
        self.starfield_layers.append(layer3_stars)
        
        # Create nebula clouds
        self.nebula_clouds = []
        for _ in range(8):  # Several nebula regions
            nebula = {
                "x": np.random.uniform(-200, 1400),
                "y": np.random.uniform(-200, 1100),
                "size": np.random.uniform(300, 800),
                "color": np.random.choice([
                    (120, 0, 180, 30),    # Purple
                    (0, 100, 180, 25),    # Blue
                    (180, 50, 120, 30),   # Pink
                    (100, 0, 150, 20),    # Deep purple
                    (0, 150, 100, 25),    # Cyan
                    (150, 80, 0, 30)      # Orange
                ]),
                "speed": np.random.uniform(0.05, 0.15),
                "rotation": np.random.uniform(0, 2 * math.pi),
                "rotation_speed": np.random.uniform(-0.002, 0.002)
            }
            self.nebula_clouds.append(nebula)
        
        # Create distant galaxies
        self.distant_galaxies = []
        for _ in range(5):  # A few distant galaxies
            galaxy = {
                "x": np.random.uniform(0, 1200),
                "y": np.random.uniform(0, 900),
                "size": np.random.uniform(80, 200),
                "brightness": np.random.randint(15, 40),
                "speed": np.random.uniform(0.02, 0.08),
                "spiral_arms": np.random.randint(2, 5),
                "rotation": np.random.uniform(0, 2 * math.pi),
                "rotation_speed": np.random.uniform(-0.001, 0.001)
            }
            self.distant_galaxies.append(galaxy)
        
        # Create space dust particles
        self.space_dust = []
        for _ in range(200):  # Fine cosmic dust
            dust = {
                "x": np.random.uniform(0, 1200),
                "y": np.random.uniform(0, 900),
                "size": np.random.uniform(0.5, 1.5),
                "brightness": np.random.randint(10, 40),
                "speed": np.random.uniform(0.3, 1.2),
                "drift_x": np.random.uniform(-0.1, 0.1),
                "drift_y": np.random.uniform(-0.1, 0.1)
            }
            self.space_dust.append(dust)

    def _update_cosmic_background(self) -> None:
        """Update cosmic background elements with parallax and animations."""
        self.cosmic_time += 0.016  # ~60fps time step
        
        if not hasattr(self, "prev_agent_position"):
            self.prev_agent_position = self.agent_position.copy()
            return
            
        # Calculate agent movement vector
        movement = self.agent_position - self.prev_agent_position
        self.prev_agent_position = self.agent_position.copy()
        
        # Update stars with enhanced parallax
        for layer in self.starfield_layers:
            for star in layer:
                # Move with parallax and zoom effect
                parallax_factor = star["speed"] * self.zoom_level
                star["x"] -= movement[0] * parallax_factor * 10
                star["y"] -= movement[1] * parallax_factor * 10
                
                # Add natural drift for cosmic movement
                star["x"] += star.get("drift_x", 0) * 0.5
                star["y"] += star.get("drift_y", 0) * 0.5
                
                # Wrap around with buffer zone
                if star["x"] < -50:
                    star["x"] = 1250
                elif star["x"] > 1250:
                    star["x"] = -50
                if star["y"] < -50:
                    star["y"] = 950
                elif star["y"] > 950:
                    star["y"] = -50
        
        # Update nebula clouds
        for nebula in self.nebula_clouds:
            # Slow parallax movement
            nebula["x"] -= movement[0] * nebula["speed"] * self.zoom_level * 2
            nebula["y"] -= movement[1] * nebula["speed"] * self.zoom_level * 2
            
            # Slow rotation
            nebula["rotation"] += nebula["rotation_speed"]
            
            # Wrap around
            if nebula["x"] < -nebula["size"]:
                nebula["x"] = 1200 + nebula["size"]
            elif nebula["x"] > 1200 + nebula["size"]:
                nebula["x"] = -nebula["size"]
            if nebula["y"] < -nebula["size"]:
                nebula["y"] = 900 + nebula["size"]
            elif nebula["y"] > 900 + nebula["size"]:
                nebula["y"] = -nebula["size"]
        
        # Update distant galaxies
        for galaxy in self.distant_galaxies:
            # Very slow parallax
            galaxy["x"] -= movement[0] * galaxy["speed"] * self.zoom_level * 1
            galaxy["y"] -= movement[1] * galaxy["speed"] * self.zoom_level * 1
            
            # Rotation
            galaxy["rotation"] += galaxy["rotation_speed"]
            
            # Wrap around
            if galaxy["x"] < -galaxy["size"]:
                galaxy["x"] = 1200 + galaxy["size"]
            elif galaxy["x"] > 1200 + galaxy["size"]:
                galaxy["x"] = -galaxy["size"]
            if galaxy["y"] < -galaxy["size"]:
                galaxy["y"] = 900 + galaxy["size"]
            elif galaxy["y"] > 900 + galaxy["size"]:
                galaxy["y"] = -galaxy["size"]
        
        # Update space dust
        for dust in self.space_dust:
            # Fast parallax movement
            dust["x"] -= movement[0] * dust["speed"] * self.zoom_level * 8
            dust["y"] -= movement[1] * dust["speed"] * self.zoom_level * 8
            
            # Natural cosmic drift
            dust["x"] += dust["drift_x"]
            dust["y"] += dust["drift_y"]
            
            # Wrap around
            if dust["x"] < -10:
                dust["x"] = 1210
            elif dust["x"] > 1210:
                dust["x"] = -10
            if dust["y"] < -10:
                dust["y"] = 910
            elif dust["y"] > 910:
                dust["y"] = -10

    def _update_zoom(self) -> None:
        """Update dynamic zoom system."""
        # Smooth zoom interpolation
        zoom_diff = self.target_zoom - self.zoom_level
        self.zoom_level += zoom_diff * self.zoom_speed
        
        # Dynamic zoom based on game state
        if hasattr(self, 'agent_energy') and self.agent_energy < 30:
            # Zoom in when energy is low for tension
            self.target_zoom = 1.3
        elif len([a for a in self.asteroid_resources if a > 0.1]) <= 2:
            # Zoom out when few asteroids remain
            self.target_zoom = 0.8
        elif hasattr(self, 'collision_flash_timer') and self.collision_flash_timer > 0:
            # Quick zoom out during collision
            self.target_zoom = 0.7
        else:
            # Normal zoom
            self.target_zoom = 1.0
        
        # Clamp zoom level
        self.zoom_level = max(0.5, min(2.0, self.zoom_level))

    def _add_timeline_event(self, event_type: str, text: str, color: tuple) -> None:
        """Add an event to the floating timeline."""
        event = {
            "type": event_type,
            "text": text,
            "color": color,
            "step": self.steps_count,
            "alpha": 255,
            "lifetime": 300  # Steps before fading
        }
        
        # Add to front of timeline
        self.event_timeline.insert(0, event)
        
        # Keep only the last N events
        if len(self.event_timeline) > self.max_timeline_events:
            self.event_timeline = self.event_timeline[:self.max_timeline_events]

    def _update_event_timeline(self) -> None:
        """Update event timeline animations."""
        # Age all events and remove expired ones
        for event in self.event_timeline[:]:  # Copy list to avoid modification during iteration
            age = self.steps_count - event["step"]
            if age > event["lifetime"]:
                self.event_timeline.remove(event)
            else:
                # Fade out over last 60 steps
                fade_start = event["lifetime"] - 60
                if age > fade_start:
                    fade_progress = (age - fade_start) / 60.0
                    event["alpha"] = int(255 * (1 - fade_progress))
                else:
                    event["alpha"] = 255

    def _process_mining_combo(self) -> None:
        """Process mining combo chain detection."""
        current_step = self.steps_count
        
        # Check if this mining action extends a combo
        if (current_step - self.combo_state["last_mining_step"]) <= self.combo_state["combo_window"]:
            self.combo_state["chain_count"] += 1
        else:
            self.combo_state["chain_count"] = 1
        
        self.combo_state["last_mining_step"] = current_step
        
        # Show combo if we have 2 or more
        if self.combo_state["chain_count"] >= 2:
            self.combo_state["display_timer"] = 120  # Show for 4 seconds at 30fps
            self.combo_state["combo_alpha"] = 255
            
            # Add special combo timeline event
            combo_text = f"x{self.combo_state['chain_count']} COMBO!"
            self._add_timeline_event("combo", combo_text, (255, 200, 0))

    def _update_combo_system(self) -> None:
        """Update combo display animations."""
        # Fade out combo display
        if self.combo_state["display_timer"] > 0:
            self.combo_state["display_timer"] -= 1
            
            # Pulsing effect
            pulse = abs(math.sin(self.steps_count * 0.3)) * 0.3 + 0.7
            self.combo_state["combo_alpha"] = int(255 * pulse)
            
            if self.combo_state["display_timer"] <= 0:
                self.combo_state["combo_alpha"] = 0
        
        # Reset combo if too much time has passed
        if (self.steps_count - self.combo_state["last_mining_step"]) > self.combo_state["combo_window"]:
            if self.combo_state["chain_count"] > 0:
                self.combo_state["chain_count"] = 0
                self.combo_state["display_timer"] = 0

    def _trigger_game_over(self, success: bool) -> None:
        """Trigger game over screen with final statistics."""
        cumulative_mining = getattr(self, "cumulative_mining_amount", 0.0)
        
        self.game_over_state = {
            "active": True,
            "fade_alpha": 0,
            "success": success,
            "final_stats": {
                "total_resources_mined": cumulative_mining,
                "resources_delivered": cumulative_mining - self.agent_inventory,
                "current_inventory": self.agent_inventory,
                "collisions": self.collision_count,
                "steps_taken": self.steps_count,
                "final_energy": self.agent_energy,
                "asteroids_depleted": len(self.asteroid_positions) - np.sum(self.asteroid_resources >= 0.1),
                "total_asteroids": len(self.asteroid_positions),
                "efficiency_score": self.compute_fitness_score()
            }
        }

__all__ = ["SpaceMining"]