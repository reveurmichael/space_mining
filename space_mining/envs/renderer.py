"""Advanced rendering system for the SpaceMining environment.

This module provides a sophisticated pygame-based renderer that creates rich
visualizations of the space mining environment. The renderer supports multiple
render modes and provides detailed visual feedback about the agent's state,
environment objects, and game statistics.

Features:
- Real-time environment visualization with smooth animations
- Detailed status displays and legends
- Adaptive color schemes based on game state
- Performance optimized rendering pipeline
- Support for headless and display modes
- Comprehensive visual feedback for debugging

The renderer is designed to be both informative for debugging and
visually appealing for demonstrations and analysis.
"""

from typing import Any, Optional, Tuple, Dict, List, Union
import warnings
from pathlib import Path

import numpy as np

# Optional pygame import with graceful fallback
try:
    import pygame
    from pygame import gfxdraw
    PYGAME_AVAILABLE = True
except ImportError:
    pygame = None
    gfxdraw = None
    PYGAME_AVAILABLE = False


class RenderingError(Exception):
    """Custom exception for rendering-related errors."""
    pass


class Renderer:
    """Advanced renderer for the SpaceMining environment.
    
    This class handles all visualization aspects of the SpaceMining environment,
    including real-time rendering, status displays, and visual feedback systems.
    The renderer is optimized for both performance and visual clarity.
    
    Attributes:
        env: Reference to the SpaceMining environment
        window: Pygame display surface or off-screen surface
        clock: Pygame clock for frame rate control
        font: Pygame font for text rendering
        screen_size: Size of the rendering window (width, height)
        scale_factor: Scaling factor for world-to-screen coordinate conversion
    """

    # Color constants for consistent theming
    COLORS = {
        'space_bg': (0, 0, 0),
        'agent_default': (50, 200, 50),
        'agent_mining': (255, 165, 0),
        'agent_carrying': (255, 255, 0),
        'mothership': (50, 150, 200),
        'asteroid_active': (200, 150, 0),
        'asteroid_depleted': (100, 100, 100),
        'asteroid_depleted_mark': (150, 150, 150),
        'obstacle': (200, 50, 50),
        'energy_bar': (0, 200, 0),
        'energy_bg': (0, 0, 0),
        'inventory_indicator': (200, 200, 0),
        'mining_beam': (255, 255, 0),
        'text_primary': (255, 255, 255),
        'text_warning': (255, 100, 100),
        'text_success': (100, 255, 100),
        'observation_radius': (100, 100, 255),
        'mining_range': (255, 0, 0),
        'health_full': (0, 255, 0),
        'health_empty': (255, 0, 0),
    }

    def __init__(self, env: Any, screen_size: Tuple[int, int] = (800, 800)) -> None:
        """Initialize the renderer.
        
        Args:
            env: SpaceMining environment instance to render.
            screen_size: Size of the rendering window as (width, height).
            
        Raises:
            ImportError: If pygame is not available.
            RenderingError: If renderer initialization fails.
        """
        if not PYGAME_AVAILABLE:
            raise ImportError(
                "pygame is required for rendering. Install with: pip install pygame"
            )
            
        self.env = env
        self.screen_size = screen_size
        self.window: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font: Optional[pygame.font.Font] = None
        self.small_font: Optional[pygame.font.Font] = None
        
        # Calculate scale factor to fit grid in screen
        self.scale_factor = min(screen_size) / max(env.grid_size, 80) * 0.8
        self.center_offset = (screen_size[0] // 2, screen_size[1] // 2)
        
        # Performance optimization flags
        self._last_render_time = 0
        self._min_render_interval = 1.0 / (env.metadata.get("render_fps", 30) * 1.2)

    def _ensure_initialized(self) -> None:
        """Ensure pygame components are initialized.
        
        Raises:
            RenderingError: If initialization fails.
        """
        if self.window is not None:
            return
            
        try:
            pygame.init()
            
            # Create display surface or off-screen surface
            if self.env.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode(self.screen_size)
                pygame.display.set_caption("Space Mining Environment")
            else:
                # Off-screen surface for rgb_array mode
                self.window = pygame.Surface(self.screen_size)
                
            # Initialize timing and fonts
            self.clock = pygame.time.Clock()
            pygame.font.init()
            
            # Load fonts with fallbacks
            font_candidates = ["Arial", "DejaVu Sans", "Liberation Sans"]
            self.font = self._load_best_font(font_candidates, 16)
            self.small_font = self._load_best_font(font_candidates, 12)
            
        except Exception as e:
            raise RenderingError(f"Failed to initialize renderer: {e}") from e

    def _load_best_font(self, font_names: List[str], size: int) -> pygame.font.Font:
        """Load the best available font from a list of candidates.
        
        Args:
            font_names: List of font names to try.
            size: Font size.
            
        Returns:
            Loaded pygame font.
        """
        for font_name in font_names:
            try:
                return pygame.font.SysFont(font_name, size)
            except:
                continue
        
        # Fallback to default font
        return pygame.font.Font(None, size)

    def to_screen(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates.
        
        Args:
            world_pos: World position as [x, y] array.
            
        Returns:
            Screen coordinates as (x, y) tuple.
        """
        x, y = world_pos
        # Center the grid and apply scaling
        world_center = self.env.grid_size / 2
        screen_x = int(self.center_offset[0] + (x - world_center) * self.scale_factor)
        screen_y = int(self.center_offset[1] + (y - world_center) * self.scale_factor)
        return screen_x, screen_y

    def _draw_mothership(self) -> None:
        """Draw the mothership with enhanced visual effects."""
        mothership_pos = self.to_screen(self.env.mothership_pos)
        base_radius = int(15 * self.scale_factor / 8.0)
        
        # Draw pulsing effect
        pulse_radius = base_radius + int(3 * np.sin(pygame.time.get_ticks() * 0.005))
        gfxdraw.filled_circle(
            self.window, 
            mothership_pos[0], 
            mothership_pos[1], 
            pulse_radius, 
            (*self.COLORS['mothership'], 128)
        )
        
        # Draw main body
        gfxdraw.filled_circle(
            self.window, 
            mothership_pos[0], 
            mothership_pos[1], 
            base_radius, 
            self.COLORS['mothership']
        )
        
        # Draw outline
        gfxdraw.aacircle(
            self.window,
            mothership_pos[0],
            mothership_pos[1],
            base_radius,
            self.COLORS['text_primary']
        )

    def _draw_asteroids(self) -> None:
        """Draw asteroids with health bars and visual effects."""
        for i, pos in enumerate(self.env.asteroid_positions):
            asteroid_pos = self.to_screen(pos)
            resource_amount = self.env.asteroid_resources[i]
            
            if resource_amount < 0.1:  # Depleted asteroid
                self._draw_depleted_asteroid(asteroid_pos)
            else:
                self._draw_active_asteroid(asteroid_pos, resource_amount, i)

    def _draw_depleted_asteroid(self, screen_pos: Tuple[int, int]) -> None:
        """Draw a depleted asteroid with visual indicators.
        
        Args:
            screen_pos: Screen position to draw at.
        """
        base_radius = int(8 * self.scale_factor / 8.0)
        
        # Draw depleted asteroid body
        gfxdraw.filled_circle(
            self.window,
            screen_pos[0],
            screen_pos[1],
            base_radius,
            self.COLORS['asteroid_depleted']
        )
        
        # Draw X mark
        mark_size = 5
        pygame.draw.line(
            self.window,
            self.COLORS['asteroid_depleted_mark'],
            (screen_pos[0] - mark_size, screen_pos[1] - mark_size),
            (screen_pos[0] + mark_size, screen_pos[1] + mark_size),
            2,
        )
        pygame.draw.line(
            self.window,
            self.COLORS['asteroid_depleted_mark'],
            (screen_pos[0] + mark_size, screen_pos[1] - mark_size),
            (screen_pos[0] - mark_size, screen_pos[1] + mark_size),
            2,
        )

    def _draw_active_asteroid(self, screen_pos: Tuple[int, int], resource_amount: float, index: int) -> None:
        """Draw an active asteroid with resources and health bar.
        
        Args:
            screen_pos: Screen position to draw at.
            resource_amount: Amount of resources in asteroid.
            index: Asteroid index for identification.
        """
        # Calculate visual properties based on resources
        base_size = 5 * self.scale_factor / 8.0
        size = int(base_size + resource_amount / 10)
        color_intensity = min(255, int(100 + resource_amount * 3))
        
        # Draw asteroid body with glow effect
        glow_radius = size + 3
        gfxdraw.filled_circle(
            self.window,
            screen_pos[0],
            screen_pos[1],
            glow_radius,
            (*self.COLORS['asteroid_active'], 64)
        )
        
        gfxdraw.filled_circle(
            self.window,
            screen_pos[0],
            screen_pos[1],
            size,
            (color_intensity, color_intensity // 2, 0),
        )
        
        # Draw health bar
        self._draw_health_bar(screen_pos, resource_amount, size)
        
        # Draw asteroid ID for debugging (small text)
        if hasattr(self, 'small_font') and self.small_font:
            id_text = self.small_font.render(str(index), True, self.COLORS['text_primary'])
            self.window.blit(id_text, (screen_pos[0] - 5, screen_pos[1] + size + 5))

    def _draw_health_bar(self, screen_pos: Tuple[int, int], resource_amount: float, asteroid_size: int) -> None:
        """Draw health bar for an asteroid.
        
        Args:
            screen_pos: Screen position of the asteroid.
            resource_amount: Current resource amount.
            asteroid_size: Visual size of the asteroid.
        """
        health_ratio = resource_amount / self.env.max_resource_per_asteroid
        health_width = int(20 * self.scale_factor / 8.0)
        health_height = int(4 * self.scale_factor / 8.0)
        health_x = screen_pos[0] - health_width // 2
        health_y = screen_pos[1] - asteroid_size - 10
        
        # Background
        pygame.draw.rect(
            self.window, 
            self.COLORS['asteroid_depleted'], 
            (health_x, health_y, health_width, health_height)
        )
        
        # Health bar with color gradient
        if health_ratio > 0:
            filled_width = int(health_width * health_ratio)
            # Color interpolation from red to green
            red = int(255 * (1 - health_ratio))
            green = int(255 * health_ratio)
            health_color = (red, green, 0)
            
            pygame.draw.rect(
                self.window, 
                health_color, 
                (health_x, health_y, filled_width, health_height)
            )

    def _draw_obstacles(self) -> None:
        """Draw moving obstacles with trail effects."""
        for i, pos in enumerate(self.env.obstacle_positions):
            obstacle_pos = self.to_screen(pos)
            radius = int(7 * self.scale_factor / 8.0)
            
            # Draw trail effect based on velocity
            if hasattr(self.env, 'obstacle_velocities'):
                velocity = self.env.obstacle_velocities[i]
                trail_length = int(np.linalg.norm(velocity) * 100)
                if trail_length > 0:
                    trail_end = self.to_screen(pos - velocity * 5)
                    pygame.draw.line(
                        self.window,
                        (*self.COLORS['obstacle'], 128),
                        trail_end,
                        obstacle_pos,
                        max(2, radius // 2)
                    )
            
            # Draw obstacle body
            gfxdraw.filled_circle(
                self.window, 
                obstacle_pos[0], 
                obstacle_pos[1], 
                radius, 
                self.COLORS['obstacle']
            )

    def _draw_agent(self) -> None:
        """Draw the agent with state-dependent visual effects."""
        agent_pos = self.to_screen(self.env.agent_position)
        base_radius = int(12 * self.scale_factor / 8.0)
        
        # Determine agent color based on state
        agent_color = self.COLORS['agent_default']
        if hasattr(self.env, "mining_asteroid_id"):
            agent_color = self.COLORS['agent_mining']
        elif self.env.agent_inventory > 0:
            agent_color = self.COLORS['agent_carrying']
        
        # Draw agent body with glow effect
        glow_radius = base_radius + 3
        gfxdraw.filled_circle(
            self.window,
            agent_pos[0],
            agent_pos[1],
            glow_radius,
            (*agent_color, 64)
        )
        
        gfxdraw.filled_circle(
            self.window,
            agent_pos[0],
            agent_pos[1],
            base_radius,
            agent_color,
        )
        
        # Draw agent outline
        gfxdraw.aacircle(
            self.window,
            agent_pos[0],
            agent_pos[1],
            base_radius,
            self.COLORS['text_primary'],
        )
        
        # Draw directional indicator based on velocity
        if np.linalg.norm(self.env.agent_velocity) > 0.1:
            self._draw_velocity_indicator(agent_pos, self.env.agent_velocity)
        
        # Draw inventory indicator
        if self.env.agent_inventory > 0:
            inventory_radius = int(5 + self.env.agent_inventory / 5)
            gfxdraw.filled_circle(
                self.window,
                agent_pos[0],
                agent_pos[1],
                inventory_radius,
                self.COLORS['inventory_indicator'],
            )
        
        # Draw energy bar
        self._draw_energy_bar(agent_pos)
        
        # Draw range indicators
        self._draw_range_indicators(agent_pos)

    def _draw_velocity_indicator(self, agent_pos: Tuple[int, int], velocity: np.ndarray) -> None:
        """Draw velocity direction indicator.
        
        Args:
            agent_pos: Agent's screen position.
            velocity: Agent's velocity vector.
        """
        vel_norm = velocity / (np.linalg.norm(velocity) + 1e-8)
        arrow_length = 20
        arrow_end = (
            int(agent_pos[0] + vel_norm[0] * arrow_length),
            int(agent_pos[1] + vel_norm[1] * arrow_length)
        )
        
        pygame.draw.line(
            self.window,
            self.COLORS['text_primary'],
            agent_pos,
            arrow_end,
            3
        )

    def _draw_energy_bar(self, agent_pos: Tuple[int, int]) -> None:
        """Draw agent energy bar.
        
        Args:
            agent_pos: Agent's screen position.
        """
        bar_width = int(40 * self.scale_factor / 8.0)
        bar_height = int(5 * self.scale_factor / 8.0)
        bar_x = agent_pos[0] - bar_width // 2
        bar_y = agent_pos[1] - int(25 * self.scale_factor / 8.0)
        
        # Background
        pygame.draw.rect(
            self.window, 
            self.COLORS['energy_bg'], 
            (bar_x, bar_y, bar_width, bar_height)
        )
        
        # Energy level
        energy_ratio = max(0, self.env.agent_energy / 150.0)
        filled_width = int(bar_width * energy_ratio)
        
        # Color based on energy level
        if energy_ratio > 0.6:
            energy_color = self.COLORS['energy_bar']
        elif energy_ratio > 0.3:
            energy_color = (255, 255, 0)  # Yellow warning
        else:
            energy_color = (255, 0, 0)  # Red critical
        
        if filled_width > 0:
            pygame.draw.rect(
                self.window, 
                energy_color, 
                (bar_x, bar_y, filled_width, bar_height)
            )

    def _draw_range_indicators(self, agent_pos: Tuple[int, int]) -> None:
        """Draw observation and mining range indicators.
        
        Args:
            agent_pos: Agent's screen position.
        """
        # Observation radius
        obs_radius = int(self.env.observation_radius * self.scale_factor)
        gfxdraw.aacircle(
            self.window,
            agent_pos[0],
            agent_pos[1],
            obs_radius,
            self.COLORS['observation_radius'],
        )
        
        # Mining range
        mining_radius = int(self.env.mining_range * self.scale_factor)
        gfxdraw.aacircle(
            self.window,
            agent_pos[0],
            agent_pos[1],
            mining_radius,
            self.COLORS['mining_range'],
        )

    def _draw_mining_beam(self) -> None:
        """Draw mining beam if agent is currently mining."""
        if hasattr(self.env, "mining_asteroid_id"):
            agent_pos = self.to_screen(self.env.agent_position)
            asteroid_pos = self.to_screen(
                self.env.asteroid_positions[self.env.mining_asteroid_id]
            )
            
            # Animated beam effect
            beam_alpha = int(128 + 64 * np.sin(pygame.time.get_ticks() * 0.01))
            beam_surface = pygame.Surface(self.screen_size, pygame.SRCALPHA)
            pygame.draw.line(
                beam_surface,
                (*self.COLORS['mining_beam'], beam_alpha),
                agent_pos,
                asteroid_pos,
                5,
            )
            self.window.blit(beam_surface, (0, 0))

    def _draw_status_display(self) -> None:
        """Draw comprehensive status information."""
        if not self.font:
            return
            
        # Gather status information
        cumulative_mining = getattr(self.env, "cumulative_mining_amount", 0.0)
        active_asteroids = np.sum(self.env.asteroid_resources >= 0.1)
        total_asteroids = len(self.env.asteroid_positions)
        
        status_lines = [
            f"Energy: {self.env.agent_energy:.0f}/150",
            f"Inventory: {self.env.agent_inventory:.0f}/{self.env.max_inventory}",
            f"Total Mined: {cumulative_mining:.1f}",
            f"Collisions: {self.env.collision_count}",
            f"Step: {self.env.steps_count}/{self.env.max_episode_steps}",
            f"Asteroids: {active_asteroids}/{total_asteroids}",
        ]
        
        # Add dynamic status based on current state
        self._add_dynamic_status(status_lines)
        
        # Render status text
        y_offset = 10
        for line in status_lines:
            # Choose color based on content
            color = self._get_status_color(line)
            text_surface = self.font.render(line, True, color)
            self.window.blit(text_surface, (10, y_offset))
            y_offset += 20

    def _add_dynamic_status(self, status_lines: List[str]) -> None:
        """Add dynamic status information based on current game state.
        
        Args:
            status_lines: List to append status lines to.
        """
        # Mining status
        if hasattr(self.env, "mining_asteroid_id"):
            status_lines.append(f"MINING: Asteroid {self.env.mining_asteroid_id}")
        elif self.env.agent_inventory > 0:
            status_lines.append("CARRYING RESOURCES")
        else:
            status_lines.append("EXPLORING")
        
        # Recent events
        current_step = self.env.steps_count
        
        # Mining event
        if (hasattr(self.env, "last_mining_info") and 
            self.env.last_mining_info.get("step", 0) == current_step):
            info = self.env.last_mining_info
            status_lines.append(
                f"MINED: {info['extracted']:.1f} from Asteroid {info['asteroid_id']}"
            )
            if info.get("asteroid_depleted", False):
                status_lines.append("ASTEROID DEPLETED!")
        
        # Delivery event
        if (hasattr(self.env, "last_delivery_info") and 
            self.env.last_delivery_info.get("step", 0) == current_step):
            delivered = self.env.last_delivery_info['delivered']
            status_lines.append(f"DELIVERED: {delivered:.1f} resources")
            status_lines.append("FULLY RECHARGED!")
        
        # Collision event
        if (hasattr(self.env, "last_collision_step") and 
            self.env.last_collision_step == current_step):
            status_lines.append("COLLISION DETECTED!")
        
        # Attempted depleted mining
        if hasattr(self.env, "tried_depleted_asteroid") and self.env.tried_depleted_asteroid:
            status_lines.append("TRIED TO MINE DEPLETED ASTEROID!")

    def _get_status_color(self, line: str) -> Tuple[int, int, int]:
        """Get appropriate color for status line.
        
        Args:
            line: Status line text.
            
        Returns:
            RGB color tuple.
        """
        line_lower = line.lower()
        if any(word in line_lower for word in ['collision', 'depleted', 'tried']):
            return self.COLORS['text_warning']
        elif any(word in line_lower for word in ['delivered', 'mined', 'recharged']):
            return self.COLORS['text_success']
        else:
            return self.COLORS['text_primary']

    def _draw_legend(self) -> None:
        """Draw legend explaining visual elements."""
        if not self.font:
            return
            
        legend_lines = [
            "LEGEND:",
            "Green Circle = Agent",
            "Blue Circle = Mothership", 
            "Red Circles = Obstacles",
            "Yellow Circles = Asteroids",
            "Gray X = Depleted Asteroids",
            "Blue Ring = Observation Range",
            "Red Ring = Mining Range",
        ]
        
        legend_x = self.screen_size[0] - 250
        legend_y = self.screen_size[1] - len(legend_lines) * 20 - 20
        
        for i, line in enumerate(legend_lines):
            color = self.COLORS['text_success'] if i == 0 else self.COLORS['text_primary']
            text_surface = self.font.render(line, True, color)
            self.window.blit(text_surface, (legend_x, legend_y + i * 18))

    def render(self) -> Optional[np.ndarray]:
        """Render the current state of the environment.
        
        Returns:
            RGB array if render_mode is 'rgb_array', None otherwise.
            
        Raises:
            RenderingError: If rendering fails.
        """
        if self.env.render_mode is None:
            return None
        
        try:
            self._ensure_initialized()
            
            # Performance throttling
            current_time = pygame.time.get_ticks() / 1000.0
            if current_time - self._last_render_time < self._min_render_interval:
                if self.env.render_mode == "rgb_array" and hasattr(self, '_last_frame'):
                    return self._last_frame
                return None
            
            self._last_render_time = current_time
            
            # Clear screen
            self.window.fill(self.COLORS['space_bg'])
            
            # Draw all environment elements
            self._draw_mothership()
            self._draw_asteroids()
            self._draw_obstacles()
            self._draw_agent()
            self._draw_mining_beam()
            
            # Draw UI elements
            self._draw_status_display()
            self._draw_legend()
            
            # Handle display update
            if self.env.render_mode == "human":
                pygame.display.flip()
                if self.clock:
                    self.clock.tick(self.env.metadata["render_fps"])
            
            elif self.env.render_mode == "rgb_array":
                frame = np.transpose(
                    np.array(pygame.surfarray.pixels3d(self.window)), 
                    axes=(1, 0, 2)
                )
                self._last_frame = frame
                return frame
                
        except Exception as e:
            raise RenderingError(f"Rendering failed: {e}") from e
        
        return None

    def close(self) -> None:
        """Close the rendering system and clean up resources."""
        try:
            if PYGAME_AVAILABLE and self.window is not None:
                if self.env.render_mode == "human":
                    pygame.display.quit()
                pygame.quit()
                
            # Clear references
            self.window = None
            self.clock = None
            self.font = None
            self.small_font = None
            
        except Exception as e:
            warnings.warn(f"Error during renderer cleanup: {e}")

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except:
            pass  # Ignore errors during destruction


__all__ = ["Renderer", "RenderingError"]
