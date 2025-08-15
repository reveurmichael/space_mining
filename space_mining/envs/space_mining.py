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
        self.shooting_stars = []  # Spectacular meteors
        self.cosmic_auroras = []  # Energy curtains
        self.pulsars = []  # Neutron stars
        self.cosmic_storms = []  # NEW: Spectacular storm systems
        self.wormholes = []  # NEW: Dimensional portals
        self.cosmic_lightning = []  # NEW: Energy discharges
        self.black_holes = []  # NEW: Massive gravitational monsters
        self.quasars = []  # NEW: Ultra-bright galactic nuclei
        self.cosmic_ribbons = []  # NEW: Flowing energy streams
        self.cosmic_time = 0.0
        
        # MASSIVE screen size for ultimate cosmic immersion
        self.window_width = 2560
        self.window_height = 1600
        
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
        """Initialize ULTIMATE cosmic background with massive scale for 2560x1600 screen."""
        # Reset cosmic time
        self.cosmic_time = 0.0
        
        # Create MASSIVE starfield layers for ultimate ultra-wide screen
        self.starfield_layers = []
        
        # Layer 1: Distant background stars (very slow, subtle)
        layer1_stars = []
        for _ in range(500):  # Massive star count for 2560x1600 screen
            star = {
                "x": np.random.uniform(0, 2560),
                "y": np.random.uniform(0, 1600),
                "size": 1,
                "brightness": np.random.randint(30, 80),
                "speed": 0.1,
                "twinkle_offset": np.random.uniform(0, 2 * math.pi),
                "twinkle_speed": np.random.uniform(0.5, 1.5),
                "color_type": np.random.choice(["white", "blue", "yellow", "red"], p=[0.5, 0.2, 0.2, 0.1])
            }
            layer1_stars.append(star)
        self.starfield_layers.append(layer1_stars)
        
        # Layer 2: Medium distance stars
        layer2_stars = []
        for _ in range(300):  # More medium stars for massive screen
            star = {
                "x": np.random.uniform(0, 2560),
                "y": np.random.uniform(0, 1600),
                "size": 2,
                "brightness": np.random.randint(60, 150),
                "speed": 0.3,
                "twinkle_offset": np.random.uniform(0, 2 * math.pi),
                "twinkle_speed": np.random.uniform(0.8, 2.0),
                "color_type": np.random.choice(["white", "blue", "yellow"], p=[0.6, 0.25, 0.15])
            }
            layer2_stars.append(star)
        self.starfield_layers.append(layer2_stars)
        
        # Layer 3: Bright foreground stars
        layer3_stars = []
        for _ in range(150):  # More bright stars for epic scale
            star = {
                "x": np.random.uniform(0, 2560),
                "y": np.random.uniform(0, 1600),
                "size": np.random.randint(3, 5),
                "brightness": np.random.randint(100, 255),
                "speed": 0.6,
                "twinkle_offset": np.random.uniform(0, 2 * math.pi),
                "twinkle_speed": np.random.uniform(1.0, 2.5),
                "color_type": np.random.choice(["white", "blue", "yellow"], p=[0.5, 0.3, 0.2])
            }
            layer3_stars.append(star)
        self.starfield_layers.append(layer3_stars)
        
        # Create SPECTACULAR nebula clouds for massive screen
        self.nebula_clouds = []
        for _ in range(24):  # More nebulae for ultimate cosmic experience
            nebula = {
                "x": np.random.uniform(-500, 3060),
                "y": np.random.uniform(-500, 2100),
                "size": np.random.uniform(600, 1600),  # Larger nebulae for bigger screen
                "inner_size": np.random.uniform(0.3, 0.7),
                "color": np.random.choice([
                    (80, 20, 150, 35),    # Deep purple
                    (20, 80, 180, 30),    # Deep blue
                    (150, 30, 100, 35),   # Magenta
                    (60, 20, 120, 25),    # Dark purple
                    (20, 120, 80, 30),    # Cyan
                    (120, 60, 20, 35),    # Orange
                    (100, 20, 150, 30),   # Violet
                    (20, 150, 120, 25)    # Teal
                ]),
                "speed": np.random.uniform(0.03, 0.12),
                "rotation": np.random.uniform(0, 2 * math.pi),
                "rotation_speed": np.random.uniform(-0.001, 0.001),
                "pulse_offset": np.random.uniform(0, 2 * math.pi),
                "pulse_speed": np.random.uniform(0.01, 0.03)
            }
            self.nebula_clouds.append(nebula)
        
        # Create MORE distant galaxies for epic cosmic scale
        self.distant_galaxies = []
        for _ in range(18):  # More galaxies for ultimate cosmic immersion
            galaxy = {
                "x": np.random.uniform(0, 2560),
                "y": np.random.uniform(0, 1600),
                "size": np.random.uniform(120, 400),  # Larger galaxies for bigger screen
                "brightness": np.random.randint(12, 35),
                "speed": np.random.uniform(0.01, 0.06),
                "spiral_arms": np.random.randint(2, 6),
                "arm_thickness": np.random.uniform(0.5, 2.0),
                "rotation": np.random.uniform(0, 2 * math.pi),
                "rotation_speed": np.random.uniform(-0.0008, 0.0008),
                "core_brightness": np.random.randint(25, 50)
            }
            self.distant_galaxies.append(galaxy)
        
        # Create MASSIVE enhanced space dust system
        self.space_dust = []
        
        # Fine cosmic dust
        for _ in range(750):  # Much more dust for massive screen
            dust = {
                "x": np.random.uniform(0, 2560),
                "y": np.random.uniform(0, 1600),
                "size": np.random.uniform(0.3, 1.0),
                "brightness": np.random.randint(8, 25),
                "speed": np.random.uniform(0.5, 2.0),
                "drift_x": np.random.uniform(-0.05, 0.05),
                "drift_y": np.random.uniform(-0.05, 0.05),
                "type": "fine"
            }
            self.space_dust.append(dust)
        
        # Larger dust particles
        for _ in range(200):  # More large dust for epic scale
            dust = {
                "x": np.random.uniform(0, 2560),
                "y": np.random.uniform(0, 1600),
                "size": np.random.uniform(1.0, 2.5),
                "brightness": np.random.randint(15, 40),
                "speed": np.random.uniform(0.8, 1.5),
                "drift_x": np.random.uniform(-0.1, 0.1),
                "drift_y": np.random.uniform(-0.1, 0.1),
                "type": "coarse"
            }
            self.space_dust.append(dust)
        
        # Create spectacular shooting stars
        self.shooting_stars = []
        for _ in range(4):  # More shooting stars for massive screen
            self._spawn_shooting_star()
        
        # Create MORE cosmic auroras for ultimate experience
        self.cosmic_auroras = []
        for _ in range(12):  # More ethereal energy curtains for massive screen
            aurora = {
                "x": np.random.uniform(-400, 2960),
                "y": np.random.uniform(-400, 2000),
                "width": np.random.uniform(200, 600),  # Larger auroras
                "height": np.random.uniform(400, 1000),
                "intensity": np.random.uniform(0.3, 0.8),
                "color": np.random.choice([
                    (0, 255, 150, 25),    # Green aurora
                    (150, 100, 255, 20),  # Purple aurora
                    (255, 150, 100, 25),  # Orange aurora
                    (100, 200, 255, 20),  # Blue aurora
                    (255, 100, 200, 25)   # Pink aurora
                ]),
                "wave_offset": np.random.uniform(0, 2 * math.pi),
                "wave_speed": np.random.uniform(0.02, 0.05),
                "drift_speed": np.random.uniform(0.01, 0.03)
            }
            self.cosmic_auroras.append(aurora)
        
        # Create MORE pulsars for cosmic grandeur
        self.pulsars = []
        for _ in range(9):  # More spectacular pulsing stars for massive screen
            pulsar = {
                "x": np.random.uniform(0, 2560),
                "y": np.random.uniform(0, 1600),
                "pulse_period": np.random.uniform(0.5, 2.0),
                "pulse_offset": np.random.uniform(0, 2 * math.pi),
                "brightness": np.random.randint(150, 255),
                "beam_length": np.random.uniform(200, 500),
                "beam_angle": np.random.uniform(0, 2 * math.pi),
                "rotation_speed": np.random.uniform(0.005, 0.02),
                "color": np.random.choice([
                    (255, 255, 255),  # White pulsar
                    (200, 255, 255),  # Blue-white
                    (255, 200, 255)   # Pink pulsar
                ])
            }
            self.pulsars.append(pulsar)
        
        # Create MASSIVE spectacular cosmic storms for ultimate scale
        self.cosmic_storms = []
        for _ in range(4):  # More massive storm systems for 2560x1600
            storm = {
                "x": np.random.uniform(0, 2560),
                "y": np.random.uniform(0, 1600),
                "size": np.random.uniform(400, 800),  # Larger storms for massive screen
                "intensity": np.random.uniform(0.4, 0.9),
                "rotation": np.random.uniform(0, 2 * math.pi),
                "rotation_speed": np.random.uniform(0.01, 0.03),
                "lightning_timer": 0.0,
                "color": np.random.choice([
                    (255, 100, 0, 40),   # Orange storm
                    (255, 0, 100, 35),   # Red storm
                    (100, 0, 255, 40),   # Blue storm
                    (255, 255, 0, 35)    # Yellow storm
                ])
            }
            self.cosmic_storms.append(storm)
        
        # Create MORE mystical wormholes for massive screen
        self.wormholes = []
        for _ in range(3):  # More dimensional portals for epic scale
            wormhole = {
                "x": np.random.uniform(300, 2260),
                "y": np.random.uniform(300, 1300),
                "size": np.random.uniform(100, 200),  # Larger wormholes
                "rotation": np.random.uniform(0, 2 * math.pi),
                "rotation_speed": np.random.uniform(0.02, 0.05),
                "pulse_offset": np.random.uniform(0, 2 * math.pi),
                "distortion_rings": 10  # More rings for epic effect
            }
            self.wormholes.append(wormhole)
        
        # Create SPECTACULAR black holes - the ultimate cosmic monsters
        self.black_holes = []
        for _ in range(2):  # Rare, massive gravitational beasts
            black_hole = {
                "x": np.random.uniform(400, 2160),
                "y": np.random.uniform(400, 1200),
                "size": np.random.uniform(60, 120),  # Event horizon size
                "accretion_disk_size": np.random.uniform(200, 400),  # Swirling matter disk
                "rotation": np.random.uniform(0, 2 * math.pi),
                "rotation_speed": np.random.uniform(0.03, 0.08),
                "gravity_rings": 12,  # Gravitational lensing effect
                "intensity": np.random.uniform(0.6, 1.0),
                "jet_length": np.random.uniform(300, 600),  # Polar jets
                "jet_angle": np.random.uniform(0, 2 * math.pi)
            }
            self.black_holes.append(black_hole)
        
        # Create ULTRA-BRIGHT quasars - galactic powerhouses
        self.quasars = []
        for _ in range(3):  # Extremely bright galactic nuclei
            quasar = {
                "x": np.random.uniform(0, 2560),
                "y": np.random.uniform(0, 1600),
                "brightness": np.random.uniform(0.8, 1.0),
                "size": np.random.uniform(40, 80),
                "beam_length": np.random.uniform(400, 800),
                "beam_width": np.random.uniform(20, 50),
                "beam_angle": np.random.uniform(0, 2 * math.pi),
                "rotation_speed": np.random.uniform(0.01, 0.03),
                "pulse_period": np.random.uniform(1.0, 3.0),
                "pulse_offset": np.random.uniform(0, 2 * math.pi),
                "color": np.random.choice([
                    (255, 255, 255),  # Pure white
                    (200, 255, 255),  # Blue-white
                    (255, 255, 200),  # Yellow-white
                    (255, 200, 255)   # Purple-white
                ])
            }
            self.quasars.append(quasar)
        
        # Create FLOWING cosmic ribbons - energy streams across space
        self.cosmic_ribbons = []
        for _ in range(4):  # Flowing energy streams
            ribbon = {
                "points": [],  # List of control points for the ribbon
                "width": np.random.uniform(15, 35),
                "length": np.random.uniform(600, 1200),
                "flow_speed": np.random.uniform(2.0, 5.0),
                "wave_frequency": np.random.uniform(0.01, 0.03),
                "wave_amplitude": np.random.uniform(50, 150),
                "color": np.random.choice([
                    (100, 255, 200, 60),  # Cyan ribbon
                    (255, 150, 100, 55),  # Orange ribbon
                    (200, 100, 255, 60),  # Purple ribbon
                    (255, 255, 100, 55)   # Yellow ribbon
                ]),
                "flow_offset": 0.0
            }
            # Generate ribbon control points
            start_x = np.random.uniform(0, 2560)
            start_y = np.random.uniform(0, 1600)
            for i in range(8):  # 8 control points for smooth curves
                point_x = start_x + (i * ribbon["length"] / 7) + np.random.uniform(-100, 100)
                point_y = start_y + np.random.uniform(-200, 200)
                ribbon["points"].append([point_x, point_y])
            self.cosmic_ribbons.append(ribbon)
        
        # Initialize cosmic lightning system
        self.cosmic_lightning = []

    def _spawn_shooting_star(self) -> None:
        """Spawn a spectacular shooting star for massive 2560x1600 screen."""
        # Random entry point from screen edge
        side = np.random.choice(['top', 'left', 'right', 'bottom'])
        
        if side == 'top':
            start_x = np.random.uniform(0, 2560)
            start_y = -50
            end_x = np.random.uniform(0, 2560)
            end_y = 1650
        elif side == 'left':
            start_x = -50
            start_y = np.random.uniform(0, 1600)
            end_x = 2610
            end_y = np.random.uniform(0, 1600)
        elif side == 'right':
            start_x = 2610
            start_y = np.random.uniform(0, 1600)
            end_x = -50
            end_y = np.random.uniform(0, 1600)
        else:  # bottom
            start_x = np.random.uniform(0, 2560)
            start_y = 1650
            end_x = np.random.uniform(0, 2560)
            end_y = -50
        
        shooting_star = {
            "x": start_x,
            "y": start_y,
            "target_x": end_x,
            "target_y": end_y,
            "speed": np.random.uniform(8.0, 15.0),
            "brightness": np.random.randint(200, 255),
            "tail_length": np.random.randint(8, 20),
            "color": np.random.choice([
                (255, 255, 255),  # White
                (255, 255, 200),  # Yellow-white
                (200, 255, 255),  # Blue-white
                (255, 200, 150)   # Orange-white
            ]),
            "lifetime": np.random.uniform(3.0, 6.0),
            "age": 0.0
        }
        self.shooting_stars.append(shooting_star)

    def _spawn_cosmic_lightning(self, storm) -> None:
        """Spawn spectacular cosmic lightning from storm systems."""
        # Create branching lightning bolt
        for _ in range(np.random.randint(1, 4)):  # 1-3 bolts per discharge
            lightning = {
                "start_x": storm["x"] + np.random.uniform(-storm["size"]/3, storm["size"]/3),
                "start_y": storm["y"] + np.random.uniform(-storm["size"]/3, storm["size"]/3),
                "end_x": storm["x"] + np.random.uniform(-storm["size"], storm["size"]),
                "end_y": storm["y"] + np.random.uniform(-storm["size"], storm["size"]),
                "intensity": np.random.uniform(0.7, 1.0),
                "color": np.random.choice([
                    (255, 255, 255),  # White lightning
                    (150, 150, 255),  # Blue lightning  
                    (255, 255, 150),  # Yellow lightning
                    (255, 150, 255)   # Purple lightning
                ]),
                "thickness": np.random.randint(2, 5),
                "lifetime": np.random.uniform(0.1, 0.3),
                "age": 0.0,
                "branches": []
            }
            
            # Add lightning branches
            for _ in range(np.random.randint(0, 3)):
                mid_x = (lightning["start_x"] + lightning["end_x"]) / 2 + np.random.uniform(-50, 50)
                mid_y = (lightning["start_y"] + lightning["end_y"]) / 2 + np.random.uniform(-50, 50)
                branch_end_x = mid_x + np.random.uniform(-100, 100)
                branch_end_y = mid_y + np.random.uniform(-100, 100)
                
                lightning["branches"].append({
                    "start_x": mid_x,
                    "start_y": mid_y,
                    "end_x": branch_end_x,
                    "end_y": branch_end_y
                })
            
            self.cosmic_lightning.append(lightning)

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
                star["x"] -= movement[0] * parallax_factor * 12
                star["y"] -= movement[1] * parallax_factor * 12
                
                # Add subtle natural drift
                star["x"] += star.get("drift_x", 0) * 0.3
                star["y"] += star.get("drift_y", 0) * 0.3
                
                # Wrap around with buffer zone for ultra-wide screen
                if star["x"] < -80:
                    star["x"] = 2000
                elif star["x"] > 2000:
                    star["x"] = -80
                if star["y"] < -80:
                    star["y"] = 1280
                elif star["y"] > 1280:
                    star["y"] = -80
        
        # Update nebula clouds
        for nebula in self.nebula_clouds:
            # Slow parallax movement
            nebula["x"] -= movement[0] * nebula["speed"] * self.zoom_level * 2
            nebula["y"] -= movement[1] * nebula["speed"] * self.zoom_level * 2
            
            # Slow rotation
            nebula["rotation"] += nebula["rotation_speed"]
            
            # Wrap around for ultra-wide screen
            if nebula["x"] < -nebula["size"]:
                nebula["x"] = 1920 + nebula["size"]
            elif nebula["x"] > 1920 + nebula["size"]:
                nebula["x"] = -nebula["size"]
            if nebula["y"] < -nebula["size"]:
                nebula["y"] = 1200 + nebula["size"]
            elif nebula["y"] > 1200 + nebula["size"]:
                nebula["y"] = -nebula["size"]
        
        # Update distant galaxies
        for galaxy in self.distant_galaxies:
            # Very slow parallax
            galaxy["x"] -= movement[0] * galaxy["speed"] * self.zoom_level * 1
            galaxy["y"] -= movement[1] * galaxy["speed"] * self.zoom_level * 1
            
            # Rotation
            galaxy["rotation"] += galaxy["rotation_speed"]
            
            # Wrap around for ultra-wide screen
            if galaxy["x"] < -galaxy["size"]:
                galaxy["x"] = 1920 + galaxy["size"]
            elif galaxy["x"] > 1920 + galaxy["size"]:
                galaxy["x"] = -galaxy["size"]
            if galaxy["y"] < -galaxy["size"]:
                galaxy["y"] = 1200 + galaxy["size"]
            elif galaxy["y"] > 1200 + galaxy["size"]:
                galaxy["y"] = -galaxy["size"]
        
        # Update space dust
        for dust in self.space_dust:
            # Fast parallax movement
            dust["x"] -= movement[0] * dust["speed"] * self.zoom_level * 8
            dust["y"] -= movement[1] * dust["speed"] * self.zoom_level * 8
            
            # Natural cosmic drift
            dust["x"] += dust["drift_x"]
            dust["y"] += dust["drift_y"]
            
            # Wrap around for ultra-wide screen
            if dust["x"] < -20:
                dust["x"] = 1940
            elif dust["x"] > 1940:
                dust["x"] = -20
            if dust["y"] < -20:
                dust["y"] = 1220
            elif dust["y"] > 1220:
                dust["y"] = -20
        
        # Update spectacular shooting stars
        for shooting_star in self.shooting_stars[:]:  # Use slice to allow removal
            shooting_star["age"] += 0.016
            
            # Move towards target
            dx = shooting_star["target_x"] - shooting_star["x"]
            dy = shooting_star["target_y"] - shooting_star["y"]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance > 0:
                shooting_star["x"] += (dx / distance) * shooting_star["speed"]
                shooting_star["y"] += (dy / distance) * shooting_star["speed"]
            
            # Remove if lifetime exceeded or reached target
            if shooting_star["age"] > shooting_star["lifetime"] or distance < 20:
                self.shooting_stars.remove(shooting_star)
        
        # Occasionally spawn new shooting stars
        if np.random.random() < 0.002:  # Very rare
            if len(self.shooting_stars) < 5:  # Limit number
                self._spawn_shooting_star()
        
        # Update cosmic auroras
        for aurora in self.cosmic_auroras:
            # Slow parallax movement
            aurora["x"] -= movement[0] * aurora["drift_speed"] * self.zoom_level * 0.5
            aurora["y"] -= movement[1] * aurora["drift_speed"] * self.zoom_level * 0.5
            
            # Update wave animation
            aurora["wave_offset"] += aurora["wave_speed"]
            
            # Wrap around for ultra-wide screen
            if aurora["x"] < -aurora["width"] - 300:
                aurora["x"] = 1920 + 300
            elif aurora["x"] > 1920 + 300:
                aurora["x"] = -aurora["width"] - 300
            if aurora["y"] < -aurora["height"] - 300:
                aurora["y"] = 1200 + 300
            elif aurora["y"] > 1200 + 300:
                aurora["y"] = -aurora["height"] - 300
        
        # Update pulsars
        for pulsar in self.pulsars:
            # Very slow parallax
            pulsar["x"] -= movement[0] * 0.01 * self.zoom_level
            pulsar["y"] -= movement[1] * 0.01 * self.zoom_level
            
            # Rotate beam
            pulsar["beam_angle"] += pulsar["rotation_speed"]
            
            # Wrap around for ultra-wide screen
            if pulsar["x"] < -100:
                pulsar["x"] = 2020
            elif pulsar["x"] > 2020:
                pulsar["x"] = -100
            if pulsar["y"] < -100:
                pulsar["y"] = 1300
            elif pulsar["y"] > 1300:
                pulsar["y"] = -100
        
        # Update spectacular cosmic storms
        for storm in self.cosmic_storms:
            # Slow drift
            storm["x"] -= movement[0] * 0.005 * self.zoom_level
            storm["y"] -= movement[1] * 0.005 * self.zoom_level
            
            # Rotation
            storm["rotation"] += storm["rotation_speed"]
            
            # Lightning generation
            storm["lightning_timer"] += 0.016
            if storm["lightning_timer"] > np.random.uniform(0.5, 2.0):
                storm["lightning_timer"] = 0.0
                # Spawn lightning from storm
                self._spawn_cosmic_lightning(storm)
            
            # Wrap around for MASSIVE 2560x1600 screen
            if storm["x"] < -storm["size"]:
                storm["x"] = 2560 + storm["size"]
            elif storm["x"] > 2560 + storm["size"]:
                storm["x"] = -storm["size"]
            if storm["y"] < -storm["size"]:
                storm["y"] = 1600 + storm["size"]
            elif storm["y"] > 1600 + storm["size"]:
                storm["y"] = -storm["size"]
        
        # Update mystical wormholes
        for wormhole in self.wormholes:
            # Minimal drift
            wormhole["x"] -= movement[0] * 0.002 * self.zoom_level
            wormhole["y"] -= movement[1] * 0.002 * self.zoom_level
            
            # Rotation
            wormhole["rotation"] += wormhole["rotation_speed"]
            
            # Wrap around for MASSIVE 2560x1600 screen
            if wormhole["x"] < -wormhole["size"]:
                wormhole["x"] = 2560 + wormhole["size"]
            elif wormhole["x"] > 2560 + wormhole["size"]:
                wormhole["x"] = -wormhole["size"]
            if wormhole["y"] < -wormhole["size"]:
                wormhole["y"] = 1600 + wormhole["size"]
            elif wormhole["y"] > 1600 + wormhole["size"]:
                wormhole["y"] = -wormhole["size"]
        
        # Update black holes
        for black_hole in self.black_holes:
            # Minimal drift
            black_hole["x"] -= movement[0] * 0.001 * self.zoom_level
            black_hole["y"] -= movement[1] * 0.001 * self.zoom_level
            
            # Rotation
            black_hole["rotation"] += black_hole["rotation_speed"]
            
            # Wrap around for MASSIVE 2560x1600 screen
            if black_hole["x"] < -black_hole["size"]:
                black_hole["x"] = 2560 + black_hole["size"]
            elif black_hole["x"] > 2560 + black_hole["size"]:
                black_hole["x"] = -black_hole["size"]
            if black_hole["y"] < -black_hole["size"]:
                black_hole["y"] = 1600 + black_hole["size"]
            elif black_hole["y"] > 1600 + black_hole["size"]:
                black_hole["y"] = -black_hole["size"]
        
        # Update quasars
        for quasar in self.quasars:
            # Very slow parallax
            quasar["x"] -= movement[0] * 0.005 * self.zoom_level
            quasar["y"] -= movement[1] * 0.005 * self.zoom_level
            
            # Rotate beam
            quasar["beam_angle"] += quasar["rotation_speed"]
            
            # Wrap around for MASSIVE 2560x1600 screen
            if quasar["x"] < -quasar["size"]:
                quasar["x"] = 2660
            elif quasar["x"] > 2660:
                quasar["x"] = -quasar["size"]
            if quasar["y"] < -quasar["size"]:
                quasar["y"] = 1700
            elif quasar["y"] > 1700:
                quasar["y"] = -quasar["size"]
        
        # Update cosmic ribbons
        for ribbon in self.cosmic_ribbons:
            # Update wave animation
            ribbon["flow_offset"] += ribbon["wave_frequency"]
            
            # Move ribbon control points with parallax
            for point in ribbon["points"]:
                point[0] -= movement[0] * ribbon["flow_speed"] * self.zoom_level * 0.3
                point[1] -= movement[1] * ribbon["flow_speed"] * self.zoom_level * 0.3
                
                # Wrap points around screen
                if point[0] < -300:
                    point[0] = 2860
                elif point[0] > 2860:
                    point[0] = -300
                if point[1] < -300:
                    point[1] = 1900
                elif point[1] > 1900:
                    point[1] = -300
        
        # Update cosmic lightning
        for lightning in self.cosmic_lightning[:]:
            lightning["age"] += 0.016
            lightning["intensity"] *= 0.95  # Fade quickly
            
            if lightning["age"] > lightning["lifetime"] or lightning["intensity"] < 0.1:
                self.cosmic_lightning.remove(lightning)

    def _update_zoom(self) -> None:
        """Update ENHANCED dynamic zoom system for spectacular cinematography."""
        # Smooth zoom interpolation with enhanced responsiveness
        zoom_diff = self.target_zoom - self.zoom_level
        self.zoom_speed = 0.03  # Slightly faster for more dynamic feel
        self.zoom_level += zoom_diff * self.zoom_speed
        
        # ENHANCED dynamic zoom based on multiple game factors
        zoom_factors = []
        
        # Energy-based zoom (tension when low energy)
        if hasattr(self, 'agent_energy'):
            if self.agent_energy < 20:
                zoom_factors.append(1.6)  # Much closer for extreme tension
            elif self.agent_energy < 50:
                zoom_factors.append(1.3)  # Close for tension
            else:
                zoom_factors.append(1.0)  # Normal
        
        # Asteroid-based zoom (wider view when few remaining)
        remaining_asteroids = len([a for a in self.asteroid_resources if a > 0.1])
        if remaining_asteroids <= 1:
            zoom_factors.append(0.6)  # Very wide for final asteroid hunt
        elif remaining_asteroids <= 3:
            zoom_factors.append(0.75)  # Wide for endgame
        else:
            zoom_factors.append(1.0)  # Normal
        
        # Collision-based zoom (dramatic zoom out)
        if hasattr(self, 'collision_flash_timer') and self.collision_flash_timer > 0:
            zoom_factors.append(0.5)  # Extreme wide angle for impact
        
        # Speed-based zoom (zoom out when moving fast)
        if hasattr(self, 'agent_velocity'):
            speed = np.linalg.norm(self.agent_velocity)
            if speed > 8:
                zoom_factors.append(0.8)  # Pull back for high speed
            elif speed > 5:
                zoom_factors.append(0.9)  # Slightly back for medium speed
        
        # Mining-based zoom (zoom in slightly when mining)
        if hasattr(self, 'mining_asteroid_id') and self.mining_asteroid_id is not None:
            zoom_factors.append(1.15)  # Closer focus during mining
        
        # Inventory-based zoom (zoom out when carrying lots)
        if hasattr(self, 'agent_inventory') and self.agent_inventory > 30:
            zoom_factors.append(0.85)  # Pull back when heavily loaded
        
        # Calculate final target zoom (take minimum for most dramatic effect)
        if zoom_factors:
            self.target_zoom = min(zoom_factors)
        else:
            self.target_zoom = 1.0
        
        # Enhanced zoom range for more dramatic cinematography
        self.zoom_level = max(0.4, min(2.5, self.zoom_level))

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