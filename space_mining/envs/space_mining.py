"""Core SpaceMining environment implementation.

This module implements the SpaceMining environment, a sophisticated space mining
simulation built on the Gymnasium framework. The environment features realistic
physics, resource management, energy systems, and complex agent-environment
interactions.

The environment simulates a space mining scenario where an autonomous agent must:
- Navigate through space using realistic physics (thrust, drag, gravity)
- Locate and mine resources from asteroids
- Manage energy consumption and inventory capacity
- Avoid collisions with moving obstacles
- Return resources to a mothership for rewards
- Optimize efficiency across multiple objectives

Key Features:
- Realistic 2D physics simulation with inertia and drag
- Dynamic resource distribution and depletion
- Multi-objective reward system promoting exploration and efficiency
- Partial observability with configurable observation radius
- Energy management system with consumption and recharging
- Collision detection and penalty system
- Comprehensive state tracking and analytics

The environment is designed to be challenging yet learnable, with carefully
balanced parameters that encourage strategic thinking and efficient resource
management.
"""

from typing import Any, Dict, Optional, Tuple, Union, List
import warnings
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .renderer import Renderer


@dataclass
class EnvironmentConfig:
    """Configuration parameters for the SpaceMining environment.
    
    This dataclass encapsulates all configurable parameters, making it easy
    to create variants of the environment and manage parameter validation.
    """
    # Episode parameters
    max_episode_steps: int = 1200
    grid_size: int = 80
    
    # Object generation parameters
    max_asteroids: int = 12
    max_resource_per_asteroid: int = 40
    min_asteroids: int = 8
    
    # Agent parameters
    observation_radius: int = 15
    max_inventory: int = 100
    mining_range: float = 8.0
    max_obs_asteroids: int = 15
    
    # Physics parameters
    dt: float = 0.1
    mass: float = 3.0
    max_force: float = 20.0
    drag_coef: float = 0.02
    gravity_strength: float = 0.01
    
    # Energy system
    initial_energy: float = 150.0
    energy_consumption_rate: float = 0.05
    mining_energy_cost: float = 1.0
    
    # Penalty/reward parameters
    obstacle_penalty: float = -10.0
    collision_limit: int = 18
    boundary_margin: float = 5.0
    
    # Environment generation
    min_obstacles: int = 4
    max_obstacles: int = 8
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if self.max_asteroids <= 0:
            raise ValueError("max_asteroids must be positive")
        if self.observation_radius <= 0:
            raise ValueError("observation_radius must be positive")
        if self.mining_range <= 0:
            raise ValueError("mining_range must be positive")


@dataclass
class AgentState:
    """Complete state information for the mining agent."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    energy: float = 150.0
    inventory: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert agent state to numpy array for observations."""
        return np.concatenate([
            self.position, 
            self.velocity, 
            [self.energy], 
            [self.inventory]
        ])


@dataclass 
class EnvironmentState:
    """Complete state of the environment."""
    agent: AgentState = field(default_factory=AgentState)
    asteroid_positions: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    asteroid_resources: np.ndarray = field(default_factory=lambda: np.empty(0))
    obstacle_positions: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    obstacle_velocities: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    mothership_pos: np.ndarray = field(default_factory=lambda: np.array([40.0, 40.0]))
    
    # Tracking variables
    collision_count: int = 0
    steps_count: int = 0
    cumulative_mining: float = 0.0
    discovered_asteroids: set = field(default_factory=set)


class EnvironmentError(Exception):
    """Custom exception for environment-related errors."""
    pass


class SpaceMining(gym.Env):
    """Advanced Space Mining Environment.

    A sophisticated space mining simulation where an autonomous agent must efficiently
    collect resources from asteroids while managing energy, avoiding obstacles, and
    optimizing multiple objectives.

    The environment features:
    - Realistic 2D physics with thrust, drag, and gravity
    - Resource depletion and regeneration mechanics  
    - Energy management with consumption and recharging
    - Partial observability with configurable observation radius
    - Multi-objective reward system promoting strategic behavior
    - Comprehensive state tracking and analytics

    Action Space:
        Box(3): [thrust_x, thrust_y, mine_action]
        - thrust_x, thrust_y: Thrust forces in x and y directions [-1, 1]
        - mine_action: Mining action [0, 1], >0.5 triggers mining

    Observation Space:
        Box(N): Flattened observation vector containing:
        - Agent state: [pos_x, pos_y, vel_x, vel_y, energy, inventory] (6 values)
        - Visible asteroids: [rel_x, rel_y, resources] * max_obs_asteroids (45 values default)
        - Mothership: [rel_x, rel_y] (2 values)

    Reward System:
        The environment implements a sophisticated multi-objective reward system:
        - Mining rewards: Proportional to resources extracted
        - Delivery rewards: Bonus for returning resources to mothership
        - Efficiency rewards: Bonuses for energy management and exploration
        - Penalty system: Collisions, boundary violations, and inefficient actions

    Episode Termination:
        Episodes terminate when:
        - Agent energy reaches zero
        - Maximum collision limit exceeded
        - All asteroids are depleted
        - Maximum episode steps reached
    """

    metadata = {
        "render_modes": ["human", "rgb_array"], 
        "render_fps": 30
    }

    def __init__(
        self,
        config: Optional[EnvironmentConfig] = None,
        render_mode: Optional[str] = None,
        **kwargs
    ) -> None:
        """Initialize the SpaceMining environment.

        Args:
            config: Environment configuration. If None, uses default configuration
                   with any overrides from kwargs.
            render_mode: Rendering mode ('human', 'rgb_array', or None).
            **kwargs: Additional configuration parameters to override defaults.

        Raises:
            ValueError: If configuration parameters are invalid.
            EnvironmentError: If environment initialization fails.
        """
        super().__init__()
        
        # Initialize configuration
        if config is None:
            # Create config from kwargs with defaults
            config_dict = {
                'max_episode_steps': kwargs.get('max_episode_steps', 1200),
                'grid_size': kwargs.get('grid_size', 80),
                'max_asteroids': kwargs.get('max_asteroids', 12),
                'max_resource_per_asteroid': kwargs.get('max_resource_per_asteroid', 40),
                'observation_radius': kwargs.get('observation_radius', 15),
                'render_mode': render_mode,
            }
            # Add other kwargs that match config fields
            for key, value in kwargs.items():
                if hasattr(EnvironmentConfig, key):
                    config_dict[key] = value
                    
            try:
                self.config = EnvironmentConfig(**config_dict)
            except Exception as e:
                raise EnvironmentError(f"Invalid configuration: {e}") from e
        else:
            self.config = config

        self.render_mode = render_mode

        # Initialize action and observation spaces
        self._setup_spaces()
        
        # Initialize environment state
        self.state = EnvironmentState()
        
        # Initialize renderer
        try:
            self.renderer = Renderer(self)
        except Exception as e:
            warnings.warn(f"Renderer initialization failed: {e}")
            self.renderer = None

        # Event tracking for detailed feedback
        self._reset_event_tracking()

    def _setup_spaces(self) -> None:
        """Initialize action and observation spaces."""
        # Action space: [thrust_x, thrust_y, mine_action]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        # Observation space calculation
        agent_state_dim = 6  # [pos_x, pos_y, vel_x, vel_y, energy, inventory]
        asteroids_dim = self.config.max_obs_asteroids * 3  # [rel_x, rel_y, resources] per asteroid
        mothership_dim = 2  # [rel_x, rel_y]
        
        total_dim = agent_state_dim + asteroids_dim + mothership_dim

        self.observation_space = spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(total_dim,),
            dtype=np.float32,
        )

    def _reset_event_tracking(self) -> None:
        """Reset event tracking variables."""
        self.last_mining_info: Optional[Dict[str, Any]] = None
        self.last_delivery_info: Optional[Dict[str, Any]] = None
        self.last_collision_step: int = -1
        self.tried_depleted_asteroid: bool = False
        self.mining_asteroid_id: Optional[int] = None

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options for environment reset.

        Returns:
            Tuple of (initial_observation, info_dict).

        Raises:
            EnvironmentError: If reset fails.
        """
        super().reset(seed=seed)
        
        try:
            # Reset state tracking
            self.state = EnvironmentState()
            self.state.steps_count = 0
            self.state.collision_count = 0
            self._reset_event_tracking()

            # Initialize agent
            self._initialize_agent()
            
            # Generate environment objects
            self._generate_asteroids()
            self._generate_obstacles()
            
            # Set mothership position (center of grid)
            self.state.mothership_pos = np.array([
                self.config.grid_size / 2, 
                self.config.grid_size / 2
            ])

            # Generate initial observation
            observation = self._get_observation()
            
            # Create info dictionary
            info = self._get_info()
            info.update({
                "episode_length": 0,
                "total_resources_available": np.sum(self.state.asteroid_resources),
                "energy_remaining": self.state.agent.energy,
            })

            return observation, info
            
        except Exception as e:
            raise EnvironmentError(f"Environment reset failed: {e}") from e

    def _initialize_agent(self) -> None:
        """Initialize agent state."""
        # Random initial position away from edges
        margin = self.config.boundary_margin * 2
        self.state.agent.position = self.np_random.uniform(
            low=margin, 
            high=self.config.grid_size - margin, 
            size=2
        )
        
        self.state.agent.velocity = np.zeros(2, dtype=np.float32)
        self.state.agent.energy = self.config.initial_energy
        self.state.agent.inventory = 0.0

    def _generate_asteroids(self) -> None:
        """Generate asteroids with resources."""
        # Determine number of asteroids
        min_asteroids = min(self.config.min_asteroids, self.config.max_asteroids)
        max_asteroids_clamped = min(12, self.config.max_asteroids) + 1
        
        if min_asteroids >= max_asteroids_clamped:
            num_asteroids = max(6, self.config.max_asteroids)
        else:
            num_asteroids = self.np_random.integers(min_asteroids, max_asteroids_clamped)

        # Generate positions (avoid edges and mothership area)
        margin = 15
        positions = []
        mothership_pos = np.array([self.config.grid_size / 2, self.config.grid_size / 2])
        
        for _ in range(num_asteroids):
            attempts = 0
            while attempts < 100:  # Prevent infinite loops
                pos = self.np_random.uniform(
                    low=margin, 
                    high=self.config.grid_size - margin, 
                    size=2
                )
                
                # Ensure minimum distance from mothership
                if np.linalg.norm(pos - mothership_pos) > 20:
                    # Ensure minimum distance from other asteroids
                    if not positions or all(np.linalg.norm(pos - p) > 10 for p in positions):
                        positions.append(pos)
                        break
                        
                attempts += 1
            
            if attempts >= 100:
                # Fallback position
                angle = len(positions) * 2 * np.pi / num_asteroids
                radius = 30
                pos = mothership_pos + radius * np.array([np.cos(angle), np.sin(angle)])
                positions.append(pos)

        self.state.asteroid_positions = np.array(positions)
        
        # Generate resource amounts
        self.state.asteroid_resources = self.np_random.uniform(
            low=25, 
            high=self.config.max_resource_per_asteroid, 
            size=len(positions)
        )

    def _generate_obstacles(self) -> None:
        """Generate moving obstacles."""
        num_obstacles = self.np_random.integers(
            self.config.min_obstacles, 
            self.config.max_obstacles + 1
        )
        
        # Generate obstacle positions (avoid mothership and asteroid areas)
        positions = []
        mothership_pos = np.array([self.config.grid_size / 2, self.config.grid_size / 2])
        
        for _ in range(num_obstacles):
            attempts = 0
            while attempts < 50:
                pos = self.np_random.uniform(
                    low=20, 
                    high=self.config.grid_size - 20, 
                    size=2
                )
                
                # Avoid mothership area
                if np.linalg.norm(pos - mothership_pos) > 25:
                    # Avoid asteroid areas
                    min_dist_to_asteroids = min(
                        [np.linalg.norm(pos - ast_pos) for ast_pos in self.state.asteroid_positions],
                        default=float('inf')
                    )
                    if min_dist_to_asteroids > 15:
                        positions.append(pos)
                        break
                        
                attempts += 1
            
            if attempts >= 50:
                # Fallback: place in corners
                corner_idx = len(positions) % 4
                corners = [
                    [20, 20], [self.config.grid_size - 20, 20],
                    [20, self.config.grid_size - 20], 
                    [self.config.grid_size - 20, self.config.grid_size - 20]
                ]
                positions.append(corners[corner_idx])

        self.state.obstacle_positions = np.array(positions)
        
        # Generate obstacle velocities
        self.state.obstacle_velocities = self.np_random.uniform(
            low=-0.2, 
            high=0.2, 
            size=(len(positions), 2)
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Action array [thrust_x, thrust_y, mine_action].

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).

        Raises:
            EnvironmentError: If step execution fails.
        """
        try:
            self.state.steps_count += 1
            
            # Validate and ensure environment is properly initialized
            self._ensure_environment_initialized()
            
            # Reset event flags
            self._reset_step_events()
            
            # Parse action
            thrust = action[:2] * self.config.max_force
            mine_action = action[2] > 0.5
            
            # Check for immediate termination conditions
            terminated, reward = self._check_immediate_termination()
            if terminated:
                observation = self._get_observation()
                info = self._get_info()
                return observation, reward, terminated, False, info

            # Apply physics
            self._apply_physics(thrust)
            
            # Handle boundary collisions
            boundary_reward = self._handle_boundary_collisions()
            
            # Update energy consumption
            self._update_energy(thrust, mine_action)
            
            # Check energy depletion
            if self.state.agent.energy <= 0:
                self.state.agent.energy = 0
                observation = self._get_observation()
                info = self._get_info()
                return observation, -10.0, True, False, info
            
            # Handle obstacle collisions
            obstacle_reward, collision_terminated = self._handle_obstacle_collisions()
            
            # Handle mining
            mining_reward = self._handle_mining(mine_action)
            
            # Handle resource delivery
            delivery_reward = self._handle_delivery()
            
            # Update moving obstacles
            self._update_obstacles()
            
            # Calculate advanced rewards
            advanced_reward, reward_info = self._compute_advanced_rewards(action)
            
            # Calculate fitness score
            fitness_score = self._compute_fitness_score()
            
            # Check termination conditions
            terminated = collision_terminated or self._check_completion()
            truncated = self.state.steps_count >= self.config.max_episode_steps
            
            if truncated:
                print(f"[EPISODE END] Step {self.state.steps_count}: Time limit reached")
            
            # Combine all rewards
            total_reward = (
                boundary_reward + obstacle_reward + mining_reward + 
                delivery_reward + advanced_reward
            )
            
            # Generate observation and info
            observation = self._get_observation()
            info = self._get_info()
            info.update({
                "fitness_score": fitness_score,
                "reward_breakdown": {
                    "boundary": boundary_reward,
                    "obstacle": obstacle_reward, 
                    "mining": mining_reward,
                    "delivery": delivery_reward,
                    "advanced": advanced_reward,
                    "total": total_reward
                }
            })
            info.update(reward_info)
            
            # Handle rendering
            if self.render_mode == "human":
                self.render()
                
            return observation, total_reward, terminated, truncated, info
            
        except Exception as e:
            raise EnvironmentError(f"Step execution failed: {e}") from e

    def _ensure_environment_initialized(self) -> None:
        """Ensure environment is properly initialized."""
        required_attrs = [
            'asteroid_positions', 'asteroid_resources',
            'obstacle_positions', 'obstacle_velocities'
        ]
        
        if not all(hasattr(self.state, attr) for attr in required_attrs):
            warnings.warn("Environment not properly initialized, resetting...")
            preset_energy = self.state.agent.energy
            self.reset()
            self.state.agent.energy = preset_energy

    def _reset_step_events(self) -> None:
        """Reset per-step event flags."""
        self.tried_depleted_asteroid = False
        if hasattr(self, 'mining_asteroid_id'):
            delattr(self, 'mining_asteroid_id')

    def _check_immediate_termination(self) -> Tuple[bool, float]:
        """Check for immediate termination conditions."""
        if self.state.agent.energy <= 0:
            return True, -1.0
        return False, 0.0

    def _apply_physics(self, thrust: np.ndarray) -> None:
        """Apply physics simulation."""
        # Calculate forces
        acceleration = thrust / self.config.mass
        
        # Drag force
        drag_acceleration = -self.config.drag_coef * self.state.agent.velocity / self.config.mass
        
        # Gravity toward mothership
        to_mothership = self.state.mothership_pos - self.state.agent.position
        distance_to_mothership = np.linalg.norm(to_mothership)
        
        if distance_to_mothership > 0:
            gravity_acceleration = (to_mothership / distance_to_mothership) * self.config.gravity_strength
        else:
            gravity_acceleration = np.zeros(2)
        
        # Update velocity and position
        total_acceleration = acceleration + drag_acceleration + gravity_acceleration
        self.state.agent.velocity += total_acceleration * self.config.dt
        self.state.agent.position += self.state.agent.velocity * self.config.dt

    def _handle_boundary_collisions(self) -> float:
        """Handle collisions with environment boundaries."""
        reward = 0.0
        
        for axis in range(2):
            if self.state.agent.position[axis] < self.config.boundary_margin:
                self.state.agent.position[axis] = self.config.boundary_margin
                self.state.agent.velocity[axis] = -0.3 * self.state.agent.velocity[axis]
                reward -= 1.0
            elif self.state.agent.position[axis] > self.config.grid_size - self.config.boundary_margin:
                self.state.agent.position[axis] = self.config.grid_size - self.config.boundary_margin
                self.state.agent.velocity[axis] = -0.3 * self.state.agent.velocity[axis]
                reward -= 1.0
                
        return reward

    def _update_energy(self, thrust: np.ndarray, mining: bool) -> None:
        """Update agent energy consumption."""
        energy_used = self.config.energy_consumption_rate * 0.5
        energy_used += np.sum(np.abs(thrust)) * 0.01
        
        if mining:
            energy_used += self.config.mining_energy_cost * 0.5
            
        self.state.agent.energy -= energy_used

    def _handle_obstacle_collisions(self) -> Tuple[float, bool]:
        """Handle collisions with obstacles."""
        reward = 0.0
        terminated = False
        
        for obstacle_pos in self.state.obstacle_positions:
            distance = np.linalg.norm(self.state.agent.position - obstacle_pos)
            if distance < 1.5:
                reward += self.config.obstacle_penalty
                self.state.collision_count += 1
                self.last_collision_step = self.state.steps_count
                
                # Apply collision response
                to_obstacle = self.state.agent.position - obstacle_pos
                if np.linalg.norm(to_obstacle) > 0:
                    to_obstacle = to_obstacle / np.linalg.norm(to_obstacle)
                    self.state.agent.velocity += to_obstacle * 2.0
                
        # Check collision limit
        if self.state.collision_count >= self.config.collision_limit:
            print(f"[EPISODE END] Step {self.state.steps_count}: Too many collisions")
            terminated = True
            
        return reward, terminated

    def _handle_mining(self, mine_action: bool) -> float:
        """Handle mining actions and rewards."""
        if not mine_action or self.state.agent.energy <= 0 or self.state.agent.inventory >= self.config.max_inventory:
            if mine_action and self.state.agent.inventory >= self.config.max_inventory:
                return -0.2  # Penalty for mining when inventory full
            return 0.0
        
        reward = 0.0
        mined_something = False
        tried_depleted = False
        
        for i, asteroid_pos in enumerate(self.state.asteroid_positions):
            distance = np.linalg.norm(self.state.agent.position - asteroid_pos)
            
            if distance < self.config.mining_range:
                if self.state.asteroid_resources[i] < 0.1:
                    tried_depleted = True
                    continue
                    
                # Calculate mining amount
                mining_efficiency = 0.6
                max_possible = min(
                    self.state.asteroid_resources[i] * mining_efficiency,
                    self.config.max_inventory - self.state.agent.inventory,
                )
                
                if max_possible > 0:
                    # Update resources and inventory
                    self.state.asteroid_resources[i] -= max_possible
                    if self.state.asteroid_resources[i] < 0.1:
                        self.state.asteroid_resources[i] = 0.0
                        
                    self.state.agent.inventory += max_possible
                    self.state.cumulative_mining += max_possible
                    
                    # Store mining info
                    self.last_mining_info = {
                        "step": self.state.steps_count,
                        "asteroid_id": i,
                        "extracted": max_possible,
                        "inventory": self.state.agent.inventory,
                        "cumulative_mining": self.state.cumulative_mining,
                        "asteroid_depleted": self.state.asteroid_resources[i] <= 0,
                    }
                    
                    self.mining_asteroid_id = i
                    reward += max_possible * 8.0
                    mined_something = True
                    
                    # Apply mining physics (reduced mobility)
                    self.state.agent.velocity *= 0.8
                    break
        
        if not mined_something:
            if tried_depleted:
                reward -= 0.2
                self.tried_depleted_asteroid = True
            else:
                reward -= 0.1
                
        return reward

    def _handle_delivery(self) -> float:
        """Handle resource delivery to mothership."""
        distance_to_mothership = np.linalg.norm(
            self.state.agent.position - self.state.mothership_pos
        )
        
        if distance_to_mothership < 12.0 and self.state.agent.inventory > 0:
            delivered_amount = self.state.agent.inventory
            reward = delivered_amount * 12.0
            
            # Store delivery info
            energy_recharged = self.config.initial_energy - self.state.agent.energy
            self.last_delivery_info = {
                "step": self.state.steps_count,
                "delivered": delivered_amount,
                "energy_recharged": energy_recharged,
            }
            
            # Reset inventory and energy
            self.state.agent.inventory = 0
            self.state.agent.energy = self.config.initial_energy
            
            # Energy recharge bonus
            reward += energy_recharged * 0.5
            
            return reward
            
        return 0.0

    def _update_obstacles(self) -> None:
        """Update moving obstacle positions."""
        for i in range(len(self.state.obstacle_positions)):
            self.state.obstacle_positions[i] += self.state.obstacle_velocities[i] * self.config.dt
            
            # Bounce off boundaries
            for axis in range(2):
                if (self.state.obstacle_positions[i][axis] < 0 or 
                    self.state.obstacle_positions[i][axis] > self.config.grid_size):
                    self.state.obstacle_velocities[i][axis] *= -1

    def _check_completion(self) -> bool:
        """Check if all asteroids are depleted."""
        if np.all(self.state.asteroid_resources < 0.1):
            print(f"[EPISODE END] Step {self.state.steps_count}: All asteroids depleted")
            return True
        return False

    def _compute_advanced_rewards(self, action: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Compute advanced reward components for strategic behavior."""
        # Reward parameters
        SPEED_LIMIT = 10.0
        EFFICIENCY_THRESHOLD = 0.5
        EXPLORATION_BONUS = 3.0
        PATH_EFFICIENCY_BONUS = 2.0
        MINING_GUIDANCE_BONUS = 2.0
        DELIVERY_GUIDANCE_BONUS = 3.0
        
        # Initialize reward components
        rewards = {
            "speed_penalty": 0.0,
            "efficiency_reward": 0.0,
            "exploration_reward": 0.0,
            "path_efficiency_reward": 0.0,
            "mining_guidance_reward": 0.0,
            "delivery_guidance_reward": 0.0,
        }
        
        # Speed penalty for excessive velocity
        speed = np.linalg.norm(self.state.agent.velocity)
        if speed > SPEED_LIMIT:
            rewards["speed_penalty"] = -0.05 * (speed - SPEED_LIMIT) ** 2
        
        # Energy efficiency reward
        energy_ratio = self.state.agent.energy / self.config.initial_energy
        if energy_ratio > EFFICIENCY_THRESHOLD:
            rewards["efficiency_reward"] = 1.0 * energy_ratio
        
        # Exploration reward for discovering new asteroids
        for i, asteroid_pos in enumerate(self.state.asteroid_positions):
            if self.state.asteroid_resources[i] <= 0.1:
                continue
                
            distance = np.linalg.norm(self.state.agent.position - asteroid_pos)
            if (distance <= self.config.observation_radius and 
                i not in self.state.discovered_asteroids):
                self.state.discovered_asteroids.add(i)
                rewards["exploration_reward"] += EXPLORATION_BONUS
        
        # Path efficiency and guidance rewards
        if self.state.agent.inventory > 0:
            # Guide toward mothership when carrying resources
            distance_to_mothership = np.linalg.norm(
                self.state.agent.position - self.state.mothership_pos
            )
            if distance_to_mothership < 15.0:
                proximity_factor = 1.0 - distance_to_mothership / 15.0
                rewards["path_efficiency_reward"] = PATH_EFFICIENCY_BONUS * 2.0 * proximity_factor
                rewards["delivery_guidance_reward"] = DELIVERY_GUIDANCE_BONUS * proximity_factor
        else:
            # Guide toward nearest asteroid when inventory empty
            nearest_dist = float("inf")
            for i, asteroid_pos in enumerate(self.state.asteroid_positions):
                if self.state.asteroid_resources[i] > 0.1:
                    dist = np.linalg.norm(self.state.agent.position - asteroid_pos)
                    nearest_dist = min(nearest_dist, dist)
            
            if nearest_dist < 10.0:
                proximity_factor = 1.0 - nearest_dist / 10.0
                rewards["path_efficiency_reward"] = PATH_EFFICIENCY_BONUS * 2.0 * proximity_factor
                rewards["mining_guidance_reward"] = MINING_GUIDANCE_BONUS * proximity_factor
        
        # Emergency guidance (low energy, no resources)
        if energy_ratio < 0.3 and self.state.agent.inventory == 0:
            distance_to_mothership = np.linalg.norm(
                self.state.agent.position - self.state.mothership_pos
            )
            if distance_to_mothership < 20.0:
                rewards["delivery_guidance_reward"] += 1.0 * (1.0 - distance_to_mothership / 20.0)
        
        total_reward = sum(rewards.values())
        return total_reward, rewards

    def _compute_fitness_score(self) -> float:
        """Compute comprehensive fitness score."""
        # Component scores
        resources_collected = self.state.agent.inventory
        energy_remaining = self.state.agent.energy / 100.0
        remaining_resources = np.sum(self.state.asteroid_resources)
        total_initial_resources = self.config.max_resource_per_asteroid * self.config.max_asteroids
        resource_depletion_ratio = 1.0 - (remaining_resources / total_initial_resources)
        
        # Proximity to nearest asteroid
        nearest_asteroid_dist = float("inf")
        for i, asteroid_pos in enumerate(self.state.asteroid_positions):
            if self.state.asteroid_resources[i] >= 0.1:
                dist = np.linalg.norm(self.state.agent.position - asteroid_pos)
                nearest_asteroid_dist = min(nearest_asteroid_dist, dist)
        
        if nearest_asteroid_dist == float("inf"):
            nearest_asteroid_dist = 0
        else:
            nearest_asteroid_dist = 1.0 - min(1.0, nearest_asteroid_dist / self.config.grid_size)
        
        # Proximity to mothership (when carrying resources)
        distance_to_mothership = np.linalg.norm(self.state.agent.position - self.state.mothership_pos)
        mothership_proximity = 1.0 - min(1.0, distance_to_mothership / self.config.grid_size)
        
        # Calculate base fitness
        fitness = (
            resources_collected * 50.0 +
            energy_remaining * 300.0 +
            resource_depletion_ratio * 200.0 +
            nearest_asteroid_dist * 100.0 * (1 - resource_depletion_ratio) +
            mothership_proximity * 100.0 * (self.state.agent.inventory > 0)
        )
        
        # Bonus components
        completion_bonus = resource_depletion_ratio * 500.0
        efficiency_bonus = 0.0
        if self.state.steps_count > 0:
            efficiency_bonus = (resource_depletion_ratio / max(1, self.state.steps_count)) * 1000.0
        survival_bonus = self.state.steps_count * 0.5
        
        fitness += completion_bonus + efficiency_bonus + survival_bonus
        return fitness

    def _get_observation(self) -> np.ndarray:
        """Generate observation vector."""
        # Agent state
        agent_obs = self.state.agent.to_array()
        
        # Visible asteroids
        asteroid_obs = np.zeros((self.config.max_obs_asteroids, 3), dtype=np.float32)
        asteroid_count = 0
        
        for i, asteroid_pos in enumerate(self.state.asteroid_positions):
            if self.state.asteroid_resources[i] < 0.1:
                continue
                
            rel_pos = asteroid_pos - self.state.agent.position
            distance = np.linalg.norm(rel_pos)
            
            if distance <= self.config.observation_radius and asteroid_count < self.config.max_obs_asteroids:
                asteroid_obs[asteroid_count] = np.concatenate([
                    rel_pos, 
                    [self.state.asteroid_resources[i]]
                ])
                asteroid_count += 1
        
        # Mothership relative position
        mothership_rel_pos = self.state.mothership_pos - self.state.agent.position
        
        # Combine all observations
        observation = np.concatenate([
            agent_obs,
            asteroid_obs.flatten(),
            mothership_rel_pos
        ])
        
        return observation.astype(self.observation_space.dtype, copy=False)

    def _get_info(self) -> Dict[str, Any]:
        """Generate comprehensive info dictionary."""
        info = {
            "agent_position": self.state.agent.position.copy(),
            "agent_velocity": self.state.agent.velocity.copy(),
            "agent_energy": float(self.state.agent.energy),
            "agent_inventory": float(self.state.agent.inventory),
            "asteroid_resources": self.state.asteroid_resources.copy(),
            "mothership_pos": self.state.mothership_pos.copy(),
            "asteroid_positions": self.state.asteroid_positions.copy(),
            "collision_count": self.state.collision_count,
            "steps_count": self.state.steps_count,
            "cumulative_mining_amount": float(self.state.cumulative_mining),
        }
        
        # Add optional attributes
        if hasattr(self, 'mining_asteroid_id'):
            info["mining_asteroid_id"] = self.mining_asteroid_id
        else:
            info["mining_asteroid_id"] = None
            
        return info

    def render(self) -> Optional[np.ndarray]:
        """Render the environment.
        
        Returns:
            RGB array if render_mode is 'rgb_array', None otherwise.
        """
        if self.renderer is None:
            warnings.warn("Renderer not available")
            return None
            
        try:
            return self.renderer.render()
        except Exception as e:
            warnings.warn(f"Rendering failed: {e}")
            return None

    def close(self) -> None:
        """Close the environment and clean up resources."""
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception as e:
                warnings.warn(f"Renderer cleanup failed: {e}")

    # Properties for backward compatibility
    @property
    def agent_position(self) -> np.ndarray:
        """Agent position (backward compatibility)."""
        return self.state.agent.position

    @agent_position.setter
    def agent_position(self, value: np.ndarray) -> None:
        self.state.agent.position = value

    @property
    def agent_velocity(self) -> np.ndarray:
        """Agent velocity (backward compatibility)."""
        return self.state.agent.velocity

    @agent_velocity.setter
    def agent_velocity(self, value: np.ndarray) -> None:
        self.state.agent.velocity = value

    @property
    def agent_energy(self) -> float:
        """Agent energy (backward compatibility)."""
        return self.state.agent.energy

    @agent_energy.setter
    def agent_energy(self, value: float) -> None:
        self.state.agent.energy = value

    @property
    def agent_inventory(self) -> float:
        """Agent inventory (backward compatibility)."""
        return self.state.agent.inventory

    @agent_inventory.setter
    def agent_inventory(self, value: float) -> None:
        self.state.agent.inventory = value

    @property
    def asteroid_positions(self) -> np.ndarray:
        """Asteroid positions (backward compatibility)."""
        return self.state.asteroid_positions

    @asteroid_positions.setter
    def asteroid_positions(self, value: np.ndarray) -> None:
        self.state.asteroid_positions = value

    @property
    def asteroid_resources(self) -> np.ndarray:
        """Asteroid resources (backward compatibility)."""
        return self.state.asteroid_resources

    @asteroid_resources.setter
    def asteroid_resources(self, value: np.ndarray) -> None:
        self.state.asteroid_resources = value

    @property
    def obstacle_positions(self) -> np.ndarray:
        """Obstacle positions (backward compatibility)."""
        return self.state.obstacle_positions

    @obstacle_positions.setter
    def obstacle_positions(self, value: np.ndarray) -> None:
        self.state.obstacle_positions = value

    @property
    def obstacle_velocities(self) -> np.ndarray:
        """Obstacle velocities (backward compatibility)."""
        return self.state.obstacle_velocities

    @obstacle_velocities.setter
    def obstacle_velocities(self, value: np.ndarray) -> None:
        self.state.obstacle_velocities = value

    @property
    def mothership_pos(self) -> np.ndarray:
        """Mothership position (backward compatibility)."""
        return self.state.mothership_pos

    @mothership_pos.setter
    def mothership_pos(self, value: np.ndarray) -> None:
        self.state.mothership_pos = value

    @property
    def collision_count(self) -> int:
        """Collision count (backward compatibility)."""
        return self.state.collision_count

    @collision_count.setter
    def collision_count(self, value: int) -> None:
        self.state.collision_count = value

    @property
    def steps_count(self) -> int:
        """Steps count (backward compatibility)."""
        return self.state.steps_count

    @steps_count.setter
    def steps_count(self, value: int) -> None:
        self.state.steps_count = value

    @property
    def cumulative_mining_amount(self) -> float:
        """Cumulative mining amount (backward compatibility)."""
        return self.state.cumulative_mining

    @cumulative_mining_amount.setter
    def cumulative_mining_amount(self, value: float) -> None:
        self.state.cumulative_mining = value

    @property
    def discovered_asteroids(self) -> set:
        """Discovered asteroids (backward compatibility)."""
        return self.state.discovered_asteroids

    @discovered_asteroids.setter
    def discovered_asteroids(self, value: set) -> None:
        self.state.discovered_asteroids = value

    # Additional properties for configuration access
    @property
    def max_episode_steps(self) -> int:
        return self.config.max_episode_steps

    @property
    def grid_size(self) -> int:
        return self.config.grid_size

    @property
    def max_asteroids(self) -> int:
        return self.config.max_asteroids

    @property
    def max_resource_per_asteroid(self) -> int:
        return self.config.max_resource_per_asteroid

    @property
    def observation_radius(self) -> int:
        return self.config.observation_radius

    @property
    def max_inventory(self) -> int:
        return self.config.max_inventory

    @property
    def mining_range(self) -> float:
        return self.config.mining_range

    @property
    def max_obs_asteroids(self) -> int:
        return self.config.max_obs_asteroids

    @property
    def dt(self) -> float:
        return self.config.dt

    @property
    def mass(self) -> float:
        return self.config.mass

    @property
    def max_force(self) -> float:
        return self.config.max_force

    @property
    def drag_coef(self) -> float:
        return self.config.drag_coef

    @property
    def gravity_strength(self) -> float:
        return self.config.gravity_strength

    @property
    def obstacle_penalty(self) -> float:
        return self.config.obstacle_penalty

    @property
    def energy_consumption_rate(self) -> float:
        return self.config.energy_consumption_rate

    @property
    def mining_energy_cost(self) -> float:
        return self.config.mining_energy_cost


__all__ = ["SpaceMining", "EnvironmentConfig", "AgentState", "EnvironmentState", "EnvironmentError"]