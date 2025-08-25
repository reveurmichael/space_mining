from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .renderer import Renderer


class SpaceMining(gym.Env):
    """
    Space Mining Environment

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
        max_episode_steps: int = 2000,
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

        # Constants for physics simulation
        self.dt: float = 0.1
        self.mass: float = (
            3.0  # Further reduced mass from 5.0 to 3.0 for much faster movement
        )
        self.max_force: float = (
            20.0  # Further increased max force from 15.0 to 20.0 for much faster movement
        )
        self.drag_coef: float = (
            0.02  # Further reduced drag coefficient from 0.05 to 0.02 for much less resistance
        )
        self.gravity_strength: float = (
            0.01  # Further reduced gravity from 0.03 to 0.01 for much easier movement
        )

        self.obstacle_penalty: float = (
            -10.0
        )  # TODO: this is strange because obstacle_penalty should be put in reward function/compute_reward, not here as an attribute of the environment

        self.energy_consumption_rate: float = 0.05  # Reduce base energy consumption
        self.mining_energy_cost: float = 1.0  # Reduce mining energy consumption

        self.steps_count: int = 0
        self.mothership_pos: np.ndarray = np.array([grid_size / 2, grid_size / 2])

        # Animation and visual effects (for renderer communication)
        self.delivery_particles = []
        self.agent_trail = []  # TODO: if possible, move this to renderer; Do we still have agent trail? Seems like not.
        self.score_popups = []
        self.collision_flash_timer = 0.0  # TODO: if possible, move this to renderer
        self.screen_shake_timer = 0.0 # TODO: if possible, move this to renderer
        self.mining_beam_offset = 0.0  # TODO: if possible, move this to renderer

        # Screen size for renderer
        # TODO: this should be moved to renderer
        self.window_width = 1920  # Standard 1080p width
        self.window_height = 1080  # Standard 1080p height

        # Zoom system for gameplay
        # TODO: those zoom related attributes should be moved to renderer
        self.zoom_level = 1.0
        self.target_zoom = 1.0
        self.zoom_speed = 0.025

        # Game over screen state
        # TODO: all game over related attributes should be moved to renderer. Also, we will not have a game over screen show stuffs. Nothing will happen after game over.
        self.game_over_state = {
            "active": False,
            "fade_alpha": 0,
            "final_stats": {},
            "success": False,
        }

        self.action_space: spaces.Box = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        self.max_obs_asteroids: int = 15
        self.max_inventory: int = (
            100  # Increased from 60 to 100 for much easier mining and higher potential scores
        )
        self.mining_range: float = (
            8.0  # Increased from 5.0 to 8.0 for much easier mining
        )

        # TODO: this should be moved to renderer, or should they stay here?
        agent_state_dim: int = 6
        asteroids_dim: int = (
            self.max_obs_asteroids * 3
        )  # TODO: this should be moved to renderer, or should they stay here?
        mothership_dim: int = (
            2  # TODO: this should be moved to renderer, or should they stay here?
        )

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
        self.delivery_count = 0
        self.last_action = np.zeros(3, dtype=np.float32)

        # Reset animation data structures
        self.delivery_particles = []
        self.agent_trail = (
            []
        )  # TODO: this should be moved to renderer, or should they stay here, or should we remove this? Since we don't have agent trail anymore.
        self.score_popups = []
        self.collision_flash_timer = (
            0.0  # TODO: this should be moved to renderer, or should they stay here?
        )
        self.screen_shake_timer = (
            0.0  # TODO: this should be moved to renderer, or should they stay here?
        )
        self.mining_beam_offset = (
            0.0  # TODO: this should be moved to renderer, or should they stay here?
        )

        # TODO: this should be moved to renderer, or should they stay here, or should we remove this? Since we don't have game over screen view anymore.
        # Reset cosmic background and game over state
        self.game_over_state = {
            "active": False,
            "fade_alpha": 0,
            "final_stats": {},
            "success": False,
        }

        self.agent_position = self.np_random.uniform(
            low=0, high=self.grid_size, size=(2,)
        )
        self.agent_velocity = np.zeros(2, dtype=np.float32)
        self.agent_energy = 150.0
        self.agent_inventory = 0.0
        self.cumulative_mining_amount = 0.0
        self.discovered_asteroids = set()

        # Initialize asteroids and their resources
        # Increase number of asteroids and resources per asteroid
        min_asteroids = min(8, self.max_asteroids)  # Increased from 4 to 8
        low = min_asteroids
        high = min(12, self.max_asteroids) + 1  # Increased max from 6 to 12
        if low >= high:
            low = max(6, self.max_asteroids)
        num_asteroids = self.np_random.integers(low, high)
        self.asteroid_positions = self.np_random.uniform(
            low=15, high=self.grid_size - 15, size=(num_asteroids, 2)
        )
        # Increase resources per asteroid for higher potential score
        self.asteroid_resources = self.np_random.uniform(
            low=25, high=40, size=(num_asteroids,)  # Increased from 10-20 to 25-40
        )

        # More obstacles for challenge
        num_obstacles = self.np_random.integers(1, 3)  # Increased from 1-3 to 4-8
        self.obstacle_positions = self.np_random.uniform(
            low=20, high=self.grid_size - 20, size=(num_obstacles, 2)
        )
        # Faster obstacle movement
        self.obstacle_velocities = self.np_random.uniform(
            low=-0.2, high=0.2, size=(num_obstacles, 2)
        )

        # TODO: it seems that we don't use these variables anymore. THose variables starting with 'prev_' should be removed.
        self.prev_min_asteroid_distance = float("inf")
        self.prev_inventory = 0.0
        self.prev_energy = self.agent_energy
        self.prev_distance_to_mothership = np.linalg.norm(
            self.agent_position - self.mothership_pos
        )

        observation = self._get_observation()
        info = {
            "total_resources_collected": 0,
            "obstacle_collisions": 0,
            "energy_remaining": self.agent_energy,
        }
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.steps_count += 1
        self.last_action = action.copy()

        # Clear step-specific flags
        if hasattr(self, "tried_depleted_asteroid"):
            delattr(self, "tried_depleted_asteroid")

        thrust = action[:2] * self.max_force
        mine = action[2] > 0.5

        reward = 0.0

        # Skip if agent has no energy
        if self.agent_energy <= 0:
            terminated = True
            truncated = False
            observation = self._get_observation()
            info = self._get_info()
            return observation, reward, terminated, truncated, info

        # Apply thrust force
        acceleration = thrust / self.mass

        # Apply drag force: F_drag = -k * v
        drag_acceleration = -self.drag_coef * self.agent_velocity / self.mass

        # Apply gravity towards mothership (simplified)
        to_mothership = self.mothership_pos - self.agent_position
        distance_to_mothership = np.linalg.norm(to_mothership)
        if distance_to_mothership > 0:
            gravity_acceleration = (
                to_mothership / distance_to_mothership
            ) * self.gravity_strength
        else:
            gravity_acceleration = np.zeros(2)

        # Update velocity using Euler integration
        self.agent_velocity += (
            acceleration + drag_acceleration + gravity_acceleration
        ) * self.dt

        # Update position
        self.agent_position += self.agent_velocity * self.dt

        # Enforce boundary conditions with stronger containment
        boundary_margin = 5.0  # Keep agent away from edges
        boundary_collision = False
        for axis in range(2):  # 2D boundaries
            if self.agent_position[axis] < boundary_margin:
                self.agent_position[axis] = boundary_margin
                self.agent_velocity[axis] = (
                    -0.3 * self.agent_velocity[axis]
                )  # Stronger bounce
                boundary_collision = True
            elif self.agent_position[axis] > self.grid_size - boundary_margin:
                self.agent_position[axis] = self.grid_size - boundary_margin
                self.agent_velocity[axis] = (
                    -0.3 * self.agent_velocity[axis]
                )  # Stronger bounce
                boundary_collision = True

        # Consume energy based on actions - much more efficient
        energy_used = (
            self.energy_consumption_rate * 0.5
        )  # Reduced base energy consumption from 2.0 to 0.5
        # Additional energy for thrust
        energy_used += (
            np.sum(np.abs(thrust)) * 0.01
        )  # Reduced thrust energy consumption from 0.02 to 0.01
        # Energy for mining if performed
        if mine:
            energy_used += (
                self.mining_energy_cost * 0.5
            )  # Reduced mining energy consumption from 2.0 to 0.5
        self.agent_energy -= energy_used
        # Track energy usage for display
        if not hasattr(self, "last_energy_used"):
            self.last_energy_used = 0.0
        self.last_energy_used = energy_used

        # Check for energy depletion
        energy_depleted = False
        if self.agent_energy <= 0:
            self.agent_energy = 0
            energy_depleted = True
            terminated = True
            # TODO: no game over screen should be shown. There is no game over screen.
            self.renderer.trigger_game_over(success=False)
        else:
            terminated = False

        # Check for collisions with obstacles
        obstacle_collisions = 0
        for obstacle_pos in self.obstacle_positions:
            distance = np.linalg.norm(self.agent_position - obstacle_pos)
            if distance < 1.5:  # Stricter collision threshold
                obstacle_collisions += 1
                self.collision_count += 1
                # Track collision for display
                if not hasattr(self, "last_collision_step"):
                    self.last_collision_step = 0
                self.last_collision_step = self.steps_count
                # Trigger collision effects
                self.collision_flash_timer = 0.3  # Flash for 0.3 seconds
                self.screen_shake_timer = 0.2  # Shake for 0.2 seconds
                # Stronger collision response to push agent away
                to_obstacle = self.agent_position - obstacle_pos
                if np.linalg.norm(to_obstacle) > 0:
                    to_obstacle = to_obstacle / np.linalg.norm(to_obstacle)
                    self.agent_velocity += to_obstacle * 5.0  # Increased push force
                    # Also move agent position away from obstacle
                    self.agent_position += to_obstacle * 2.0
        # Terminate if too many collisions - reasonable tolerance
        if (
            self.collision_count >= 20 # TODO: this should be reduced to 8, if this is to be the same as DyCoT. Maybe DyCoT can change to 20 as well.
        ):  # Reduced from 20 to 8 for stricter collision control
            print(
                f"[EPISODE END] Step {self.steps_count}: Too many collisions, terminating episode."
            )
            terminated = True
            # TODO: no game over screen should be shown. There is no game over screen.
            self.renderer.trigger_game_over(success=False)

        # Enhanced mining action - much easier and more rewarding
        mining_success = False
        mining_amount = 0.0
        tried_depleted_asteroid = False

        if mine and self.agent_energy > 0 and self.agent_inventory < self.max_inventory:
            for i, asteroid_pos in enumerate(self.asteroid_positions):
                distance = np.linalg.norm(self.agent_position - asteroid_pos)
                if distance < self.mining_range:
                    if self.asteroid_resources[i] <= 0.1:
                        # Agent tried to mine a depleted asteroid
                        tried_depleted_asteroid = True
                        continue

                    # Extract resources from asteroid (1-2 mining attempts to deplete)
                    mining_efficiency = (
                        0.6  # 60% of remaining resources per mining attempt
                    )
                    max_possible = min(
                        self.asteroid_resources[i] * mining_efficiency,
                        self.max_inventory - self.agent_inventory,
                    )
                    if max_possible > 0:
                        self.asteroid_resources[i] -= max_possible
                        self.agent_inventory += max_possible
                        mining_amount = max_possible

                        # Track cumulative mining amount
                        if not hasattr(self, "cumulative_mining_amount"):
                            self.cumulative_mining_amount = 0.0
                        self.cumulative_mining_amount += max_possible

                        # Track mining for display
                        if not hasattr(self, "last_mining_info"):
                            self.last_mining_info = {}
                        self.last_mining_info = {
                            "step": self.steps_count,
                            "asteroid_id": i,
                            "extracted": max_possible,
                            "inventory": self.agent_inventory,
                            "cumulative_mining": self.cumulative_mining_amount,
                            "asteroid_depleted": self.asteroid_resources[i] <= 0.1,
                        }
                        # Set mining asteroid ID for display
                        self.mining_asteroid_id = i
                        mining_success = True
                        # Add score popup for mining
                        self.renderer.add_score_popup(
                            f"+{max_possible:.1f}", asteroid_pos.copy(), (255, 255, 0)
                        )
                        self.agent_velocity *= 0.8  # Reduced speed reduction from 0.7 to 0.8 for less slowdown
                        break

            if not mining_success:
                if tried_depleted_asteroid:
                    # Track depleted asteroid mining attempt for display
                    self.tried_depleted_asteroid = True
                # Clear mining asteroid ID if not mining
                if hasattr(self, "mining_asteroid_id"):
                    delattr(self, "mining_asteroid_id")

        elif mine and self.agent_inventory >= self.max_inventory:
            # Clear mining asteroid ID if not mining
            if hasattr(self, "mining_asteroid_id"):
                delattr(self, "mining_asteroid_id")
        elif not mine:
            # Clear mining asteroid ID if not mining
            if hasattr(self, "mining_asteroid_id"):
                delattr(self, "mining_asteroid_id")

        # Check for delivery to mothership - Much easier delivery
        delivery_success = False
        delivered_amount = 0.0
        distance_to_mothership = np.linalg.norm(
            self.agent_position - self.mothership_pos
        )
        if (
            distance_to_mothership < 12.0 and self.agent_inventory > 0
        ):  # Increased delivery range from 8.0 to 12.0
            delivered_amount = self.agent_inventory
            delivery_success = True
            self.delivery_count = getattr(self, "delivery_count", 0) + 1
            # Spawn delivery particles
            self.renderer.spawn_delivery_particles(
                self.agent_position.copy(), self.mothership_pos.copy()
            )
            # Add score popup for delivery
            self.renderer.add_score_popup(
                f"+{delivered_amount:.1f}", self.agent_position.copy(), (0, 255, 0)
            )
            # Track delivery for display
            if not hasattr(self, "last_delivery_info"):
                self.last_delivery_info = {}
            # Calculate energy recharge before setting to full
            energy_recharged = 150.0 - self.agent_energy  # Full recharge to 150.0
            self.last_delivery_info = {
                "step": self.steps_count,
                "delivered": delivered_amount,
                "energy_recharged": energy_recharged,  # Correct calculation
            }
            self.agent_inventory = 0
            # Fully recharge energy when at mothership
            self.agent_energy = 150.0  # Set to full energy

        # Update obstacles
        for i in range(len(self.obstacle_positions)):
            self.obstacle_positions[i] += self.obstacle_velocities[i] * self.dt

            # Simple boundary reflection
            for axis in range(2):  # 2D boundaries
                if (
                    self.obstacle_positions[i][axis] < 0
                    or self.obstacle_positions[i][axis] > self.grid_size
                ):
                    self.obstacle_velocities[i][axis] *= -1

        # Animations and zoom are now handled by the renderer during rendering

        observation = self._get_observation()

        # Check if episode should be truncated due to time limit
        truncated = self.steps_count >= self.max_episode_steps
        if truncated:
            # TODO: no game over screen should be shown. There is no game over screen.
            self.renderer.trigger_game_over(success=False)
            print(
                f"[EPISODE END] Step {self.steps_count}: Time limit reached ({self.max_episode_steps} steps) - Episode truncated"
            )

        # Terminate if all asteroids are depleted (exploration complete)
        if np.all(self.asteroid_resources <= 0.1):
            terminated = True
            # TODO: no game over screen should be shown. There is no game over screen.
            self.renderer.trigger_game_over(success=True)
            info = self._get_info()
            info["exploration_complete"] = True
            print(
                f"[EPISODE END] Step {self.steps_count}: All asteroids depleted - Episode completed successfully"
            )

        info = self._get_info()
        info["obstacle_collisions"] = obstacle_collisions

        # Compute all rewards in one place
        reward, reward_info = self.compute_reward(
            action,
            observation,
            info,
            boundary_collision=boundary_collision,
            obstacle_collisions=obstacle_collisions,
            mining_success=mining_success,
            mining_amount=mining_amount,
            tried_depleted_asteroid=tried_depleted_asteroid,
            delivery_success=delivery_success,
            delivered_amount=delivered_amount,
            energy_depleted=energy_depleted,
            terminated=terminated,
            truncated=truncated,
        )

        fitness_score = self.compute_fitness_score()
        info["fitness_score"] = fitness_score
        info.update(reward_info)

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        agent_state = np.concatenate(
            [
                self.agent_position,
                self.agent_velocity,
                [self.agent_energy],
                [self.agent_inventory],
            ]
        )
        asteroid_obs = np.zeros((self.max_obs_asteroids, 3), dtype=np.float32)
        asteroid_count = 0
        for i, asteroid_pos in enumerate(self.asteroid_positions):
            if self.asteroid_resources[i] < 0.1:
                continue
            rel_pos = asteroid_pos - self.agent_position
            distance = np.linalg.norm(rel_pos)
            if (
                distance <= self.observation_radius
                and asteroid_count < self.max_obs_asteroids
            ):
                asteroid_obs[asteroid_count] = np.concatenate(
                    [rel_pos, [self.asteroid_resources[i]]]
                )
                asteroid_count += 1
        mothership_rel_pos = self.mothership_pos - self.agent_position
        observation = np.concatenate(
            [agent_state, asteroid_obs.flatten(), mothership_rel_pos]
        )
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
            nearest_asteroid_dist = 1.0 - min(
                1.0, nearest_asteroid_dist / self.grid_size
            )

        distance_to_mothership = np.linalg.norm(
            self.agent_position - self.mothership_pos
        )
        mothership_proximity = 1.0 - min(1.0, distance_to_mothership / self.grid_size)

        fitness = (
            resources_collected * 50.0
            + energy_remaining * 300.0
            + resource_depletion_ratio * 200.0
            + nearest_asteroid_dist * 100.0 * (1 - resource_depletion_ratio)
            + mothership_proximity * 100.0 * (self.agent_inventory > 0)
        )
        completion_bonus = resource_depletion_ratio * 500.0
        efficiency_bonus = (
            (self.steps_count > 0)
            * (resource_depletion_ratio / max(1, self.steps_count))
            * 1000.0
        )
        survival_bonus = self.steps_count * 0.5
        fitness += completion_bonus + efficiency_bonus + survival_bonus
        return fitness

    def render(self) -> Optional[np.ndarray]:
        return self.renderer.render()

    def close(self) -> None:
        self.renderer.close()

    def compute_reward(
        self,
        action: np.ndarray,
        observation: np.ndarray,
        info: Dict[str, Any],
        **kwargs,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        KISS 改进版奖励：
        - 更高的单位采矿奖励，强烈鼓励立即把小行星挖空；
        - 增加“挖干净（depletion）”一次性奖励；
        - 增加对“总资源减少”的 shaping 奖励（鼓励去远处挖完）；
        - 强化靠近障碍的稠密惩罚和碰撞惩罚（但训练时可配合提高终止阈值）；
        - 保持简单清晰，便于调参。
        """
        import numpy as _np

        # --- Defensive init ---
        if not hasattr(self, "obstacle_collisions"):
            self.obstacle_collisions = 0
        if not hasattr(self, "mining_successes"):
            self.mining_successes = 0
        if not hasattr(self, "delivery_successes"):
            self.delivery_successes = 0
        if not hasattr(self, "discovered_asteroids"):
            self.discovered_asteroids = set()
        if not hasattr(self, "prev_remaining_resources"):
            self.prev_remaining_resources = float(_np.sum(getattr(self, "asteroid_resources", _np.array([]))))

        # --- Safe reads ---
        mining_amount = float(kwargs.get("mining_amount", 0.0))
        delivered_amount = float(kwargs.get("delivered_amount", 0.0))
        obstacle_collisions = int(kwargs.get("obstacle_collisions", 0))
        boundary_collision = int(kwargs.get("boundary_collision", 0))
        tried_depleted = bool(kwargs.get("tried_depleted_asteroid", False))

        # ===== Tunable constants (simple) =====
        MAX_ENERGY = getattr(self, "max_energy", 150.0)

        # 增强采矿激励（提高到 20 -> 激进鼓励挖完）
        UNIT_MINING_REWARD = 20.0
        UNIT_DELIVERY_REWARD = 20.0

        # 碰撞惩罚（仍然较大）
        OBSTACLE_COLLISION_PENALTY = -250.0
        BOUNDARY_COLLISION_PENALTY = -20.0

        # 靠近障碍的稠密惩罚（从更远距离开始处罚）
        RISK_THRESHOLD = 6.0
        RISK_PENALTY_COEFF = -60.0

        # 能量惩罚（小）
        ENERGY_PENALTY_COEFF = -0.4

        # 探索 / 引导
        ASTEROID_DISCOVER_BONUS = 1.0   # 发现新小行星的即时小奖励（鼓励探索）
        ASTEROID_DEPLETION_BONUS = 60.0 # 把单个小行星挖空的一次性奖励（提高，鼓励把残余挖干净）
        COMPLETION_BONUS = 500.0       # 全部清空的大额奖励

        # 对 total remaining 的 shaping（鼓励减少远处资源）
        RESOURCE_DEPLETION_SHAPING_SCALE = 1.0  # 把差额乘以这个值（比原来更有力量）

        # 时间步小惩罚
        TIME_STEP_REWARD = -0.001

        # ===== Sparse core terms =====
        mining_reward = UNIT_MINING_REWARD * mining_amount
        delivery_reward = UNIT_DELIVERY_REWARD * delivered_amount
        obstacle_collision_penalty = OBSTACLE_COLLISION_PENALTY * obstacle_collisions
        boundary_collision_penalty = BOUNDARY_COLLISION_PENALTY * boundary_collision

        # ===== Energy penalty =====
        energy_deficit_ratio = max(0.0, (MAX_ENERGY - float(self.agent_energy)) / MAX_ENERGY)
        energy_penalty = ENERGY_PENALTY_COEFF * energy_deficit_ratio
        energy_safety_bonus = 0.05 if self.agent_energy > 0.4 * MAX_ENERGY else 0.0

        # ===== Dense risk penalty: nearest obstacle distance =====
        min_obs_dist = float("inf")
        for pos in getattr(self, "obstacle_positions", []):
            d = _np.linalg.norm(self.agent_position - pos)
            if d < min_obs_dist:
                min_obs_dist = d
        risk_penalty = 0.0
        if min_obs_dist < RISK_THRESHOLD:
            risk_penalty = RISK_PENALTY_COEFF * (1.0 - (min_obs_dist / RISK_THRESHOLD))

        # ===== Discovery & proximity bonuses =====
        asteroid_proximity_reward = 0.0
        if mining_amount == 0 and delivered_amount == 0:
            min_ast_dist = float("inf")
            for idx, pos in enumerate(getattr(self, "asteroid_positions", [])):
                # only consider non-depleted asteroids
                if self.asteroid_resources[idx] > 0.1:
                    d = _np.linalg.norm(self.agent_position - pos)
                    if d < min_ast_dist:
                        min_ast_dist = d
                    # discovery bonus when newly seen
                    if d <= self.observation_radius and idx not in self.discovered_asteroids:
                        self.discovered_asteroids.add(idx)
                        asteroid_proximity_reward += ASTEROID_DISCOVER_BONUS
            if min_ast_dist < self.observation_radius:
                asteroid_proximity_reward += 0.5 * (1.0 - min_ast_dist / self.observation_radius)

        mothership_proximity_reward = 0.0
        if self.agent_inventory > 0:
            d2m = _np.linalg.norm(self.agent_position - self.mothership_pos)
            if d2m < self.observation_radius:
                mothership_proximity_reward = 0.5 * (1.0 - d2m / self.observation_radius)

        # ===== Asteroid depletion & global completion shaping =====
        asteroid_depletion_bonus = 0.0
        last_mining = getattr(self, "last_mining_info", None)
        if last_mining and last_mining.get("step", -1) == self.steps_count and last_mining.get("asteroid_depleted", False):
            asteroid_depletion_bonus = ASTEROID_DEPLETION_BONUS

        remaining_resources = float(_np.sum(getattr(self, "asteroid_resources", _np.array([]))))
        prev_remaining = getattr(self, "prev_remaining_resources", remaining_resources)
        # 强化资源减少的 shaping（scale > 0）
        resource_depletion_shaping = max(0.0, prev_remaining - remaining_resources) * RESOURCE_DEPLETION_SHAPING_SCALE
        self.prev_remaining_resources = remaining_resources

        completion_bonus = 0.0
        if remaining_resources <= 1e-6 and not getattr(self, "_completion_awarded", False):
            completion_bonus = COMPLETION_BONUS
            self._completion_awarded = True

        tried_depleted_penalty = -1.0 if tried_depleted else 0.0

        # ===== Sum up =====
        total_reward = (
            mining_reward
            + delivery_reward
            + obstacle_collision_penalty
            + boundary_collision_penalty
            + energy_penalty
            + risk_penalty
            + asteroid_proximity_reward
            + mothership_proximity_reward
            + energy_safety_bonus
            + asteroid_depletion_bonus
            + resource_depletion_shaping
            + completion_bonus
            + tried_depleted_penalty
            + TIME_STEP_REWARD
        )

        # clip for stability
        total_reward = float(_np.clip(total_reward, -2000.0, 2000.0))

        # ===== Update statistics =====
        self.obstacle_collisions += obstacle_collisions
        self.mining_successes += int(mining_amount > 0)
        self.delivery_successes += int(delivered_amount > 0)

        reward_info = {
            "mining_reward": mining_reward,
            "delivery_reward": delivery_reward,
            "obstacle_collision_penalty": obstacle_collision_penalty,
            "boundary_collision_penalty": boundary_collision_penalty,
            "energy_penalty": energy_penalty,
            "risk_penalty": risk_penalty,
            "asteroid_proximity_reward": asteroid_proximity_reward,
            "mothership_proximity_reward": mothership_proximity_reward,
            "asteroid_depletion_bonus": asteroid_depletion_bonus,
            "resource_depletion_shaping": resource_depletion_shaping,
            "completion_bonus": completion_bonus,
            "tried_depleted_penalty": tried_depleted_penalty,
            "time_step_reward": TIME_STEP_REWARD,
            "min_obs_dist": None if min_obs_dist == float("inf") else min_obs_dist,
            "remaining_resources": remaining_resources,
            "total_reward": total_reward,
        }

        return total_reward, reward_info


__all__ = ["SpaceMining"]
