from typing import Any, Optional

import numpy as np
import math


class Renderer:
    """Handles rendering for the SpaceMining environment."""

    def __init__(self, env: Any) -> None:
        self.env: Any = env
        self.window: Optional[Any] = None
        self.clock: Optional[Any] = None
        self.font: Optional[Any] = None
        
        # Cosmic background system (managed by renderer)
        self.starfield_layers = []
        self.nebula_clouds = []
        self.distant_galaxies = []
        self.space_dust = []
        self.cosmic_auroras = []
        self.cosmic_time = 0.0
        self.prev_agent_position = None
        
        # Initialize cosmic background
        self._initialize_cosmic_background()

    def _initialize_cosmic_background(self) -> None:
        """Initialize the perfect cosmic universe for 1920x1080."""
        import numpy as np
        import math
        
        self.cosmic_time = 0.0
        
        # Only starfield (small stars 1–5 px) with varied slow speeds
        self.starfield_layers = []
        
        # Layer 1: very slow, tiniest stars
        layer1_stars = []
        for _ in range(320):
            layer1_stars.append({
                "x": np.random.uniform(0, 1920),
                "y": np.random.uniform(0, 1080),
                "size": np.random.choice([1, 2], p=[0.9, 0.1]),
                "brightness": np.random.randint(15, 55),
                "speed": np.random.uniform(0.01, 0.03),  # Increased speed
                "burst_speed": np.random.uniform(0.05, 0.1) if np.random.random() < 0.1 else 0.0,  # Occasional burst
                "drift_x": np.random.uniform(-0.05, 0.05),
                "drift_y": np.random.uniform(-0.05, 0.05),
                "color_type": np.random.choice(["white", "blue", "yellow"], p=[0.85, 0.1, 0.05])
            })
        self.starfield_layers.append(layer1_stars)
        
        # Layer 2: slow, small stars
        layer2_stars = []
        for _ in range(240):
            layer2_stars.append({
                "x": np.random.uniform(0, 1920),
                "y": np.random.uniform(0, 1080),
                "size": np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1]),
                "brightness": np.random.randint(25, 80),
                "speed": np.random.uniform(0.03, 0.06),  # Increased speed
                "burst_speed": np.random.uniform(0.1, 0.2) if np.random.random() < 0.15 else 0.0,  # Occasional burst
                "drift_x": np.random.uniform(-0.07, 0.07),
                "drift_y": np.random.uniform(-0.07, 0.07),
                "color_type": np.random.choice(["white", "blue", "yellow"], p=[0.8, 0.15, 0.05])
            })
        self.starfield_layers.append(layer2_stars)
        
        # Layer 3: a bit faster, still small (up to 5 px)
        layer3_stars = []
        for _ in range(160):
            layer3_stars.append({
                "x": np.random.uniform(0, 1920),
                "y": np.random.uniform(0, 1080),
                "size": np.random.choice([2, 3, 4, 5], p=[0.5, 0.3, 0.15, 0.05]),
                "brightness": np.random.randint(35, 110),
                "speed": np.random.uniform(0.05, 0.1),  # Increased speed
                "burst_speed": np.random.uniform(0.2, 0.3) if np.random.random() < 0.2 else 0.0,  # Occasional burst
                "drift_x": np.random.uniform(-0.1, 0.1),
                "drift_y": np.random.uniform(-0.1, 0.1),
                "color_type": np.random.choice(["white", "blue", "yellow"], p=[0.75, 0.2, 0.05])
            })
        self.starfield_layers.append(layer3_stars)
        
        # Remove other cosmic elements per request
        self.nebula_clouds = []
        self.distant_galaxies = []
        self.space_dust = []
        self.cosmic_auroras = []

    def _update_cosmic_background(self) -> None:
        """Update cosmic background elements with optimized parallax for 1920x1080."""
        import numpy as np
        
        self.cosmic_time += 0.016  # ~60fps time step
        
        # Calculate agent movement for parallax
        movement = np.array([0.0, 0.0])
        if self.prev_agent_position is not None:
            movement = self.env.agent_position - self.prev_agent_position
        self.prev_agent_position = self.env.agent_position.copy()
        
        # Update stars with optimized parallax + independent drift
        for layer_idx, layer in enumerate(self.starfield_layers):
            for star in layer:
                # Subtle parallax based on layer
                current_speed = star.get("speed", 0.006)
                if star.get("burst_speed", 0.0) > 0 and np.random.random() < 0.05:  # 5% chance to trigger burst
                    current_speed += star.get("burst_speed", 0.0)
                parallax_factor = current_speed * self.env.zoom_level * (0.4 + 0.3 * layer_idx)
                star["x"] -= movement[0] * parallax_factor
                star["y"] -= movement[1] * parallax_factor
                
                # Independent very-slow drift
                star["x"] += star.get("drift_x", 0.0)
                star["y"] += star.get("drift_y", 0.0)
                
                # Wrap around screen with proper bounds for 1920x1080
                if star["x"] < -10: 
                    star["x"] = 1930
                elif star["x"] > 1930: 
                    star["x"] = -10
                if star["y"] < -10: 
                    star["y"] = 1090
                elif star["y"] > 1090: 
                    star["y"] = -10
        

    def render(self) -> Optional[np.ndarray]:
        """Render the current state of the environment."""
        if self.env.render_mode is None:
            return

        try:
            import pygame
            from pygame import gfxdraw
            import math
            import random
        except ImportError as exc:
            raise ImportError("pygame is not installed, run `pip install pygame`") from exc

        if self.window is None:
            pygame.init()
            # For headless rendering, avoid creating a window
            if self.env.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode((1920, 1080))  # Standard 1080p for optimal performance
                pygame.display.set_caption("🌌 Space Mining Universe - Perfect Harmony 🌌")
            else:
                # Off-screen surface for rgb_array mode
                self.window = pygame.Surface((1920, 1080))
            if self.clock is None:
                self.clock = pygame.time.Clock()
            if self.font is None:
                pygame.font.init()
                self.font = pygame.font.SysFont("Arial", 18)

        # Screen shake offset
        shake_offset = [0, 0]
        if self.env.screen_shake_timer > 0:
            shake_intensity = min(8, int(self.env.screen_shake_timer * 30))
            shake_offset[0] = random.randint(-shake_intensity, shake_intensity)
            shake_offset[1] = random.randint(-shake_intensity, shake_intensity)

        # Update cosmic background
        self._update_cosmic_background()
        
        # Update animations and zoom
        self.update_animations()
        self.update_zoom()

        # Perfect cosmic void - maximum atmospheric depth
        self.window.fill((0, 0, 3))  # Pure deep space

        # Draw starfield only
        self._draw_starfield()

        # Perfect coordinate transformation for 1920x1080 cosmic immersion  
        def to_screen(pos, scale=10.0):  # Perfect scale for cosmic viewing
            x, y = pos
            zoom_scale = scale * self.env.zoom_level
            screen_x = int(960 + (x - 40) * zoom_scale + shake_offset[0])  # Perfect center
            screen_y = int(540 + (y - 40) * zoom_scale + shake_offset[1])  # Perfect center
            return screen_x, screen_y

        # Draw mothership with safe zone aura
        mothership_pos_2d = to_screen(self.env.mothership_pos)
        
        # Mothership safe zone aura (pulsing blue)
        pulse_intensity = math.sin(self.env.steps_count * 0.08) * 0.3 + 0.7
        safe_zone_radius = 12 * 10  # Safe zone is 12 units in game coordinates
        
        # Create multiple aura layers for depth
        for i in range(5):
            aura_radius = safe_zone_radius - i * 15
            if aura_radius > 0:
                aura_alpha = int(40 * pulse_intensity * (1 - i * 0.15))
                if aura_alpha > 0:
                    aura_surface = pygame.Surface((aura_radius * 2 + 20, aura_radius * 2 + 20), pygame.SRCALPHA)
                    aura_color = (30, 120, 255, aura_alpha)
                    gfxdraw.filled_circle(
                        aura_surface, 
                        aura_radius + 10, 
                        aura_radius + 10, 
                        aura_radius, 
                        aura_color
                    )
                    self.window.blit(
                        aura_surface, 
                        (mothership_pos_2d[0] - aura_radius - 10, mothership_pos_2d[1] - aura_radius - 10)
                    )

        # Draw mothership with enhanced glow
        gfxdraw.filled_circle(
            self.window, mothership_pos_2d[0], mothership_pos_2d[1], 20, (30, 120, 200)
        )
        # Enhanced mothership glow effect
        for i in range(6):
            alpha = max(20, 150 - i * 25)
            if alpha > 0:
                glow_surface = pygame.Surface((40 + i * 8 + 10, 40 + i * 8 + 10), pygame.SRCALPHA)
                gfxdraw.filled_circle(glow_surface, 20 + i * 4 + 5, 20 + i * 4 + 5, 20 + i * 4, (30, 120, 200, alpha))
                self.window.blit(glow_surface, (mothership_pos_2d[0] - 20 - i * 4 - 5, mothership_pos_2d[1] - 20 - i * 4 - 5), special_flags=pygame.BLEND_ADD)

        # Draw asteroids with fixed yellow color and size-based resource indication
        for i, pos in enumerate(self.env.asteroid_positions):
            if self.env.asteroid_resources[i] < 0.1:  # Depleted asteroid
                asteroid_pos_2d = to_screen(pos)
                gfxdraw.filled_circle(
                    self.window,
                    asteroid_pos_2d[0],
                    asteroid_pos_2d[1],
                    8,
                    (80, 80, 80),
                )
                # Draw X mark for depleted asteroid
                pygame.draw.line(
                    self.window,
                    (150, 150, 150),
                    (asteroid_pos_2d[0] - 6, asteroid_pos_2d[1] - 6),
                    (asteroid_pos_2d[0] + 6, asteroid_pos_2d[1] + 6),
                    3,
                )
                pygame.draw.line(
                    self.window,
                    (150, 150, 150),
                    (asteroid_pos_2d[0] + 6, asteroid_pos_2d[1] - 6),
                    (asteroid_pos_2d[0] - 6, asteroid_pos_2d[1] + 6),
                    3,
                )
                continue

            asteroid_pos_2d = to_screen(pos)
            resource_ratio = self.env.asteroid_resources[i] / 40.0
            size = int(10 + resource_ratio * 10)  # 10-20 pixels for larger screen
            
            # Fixed yellow color for all active asteroids
            gfxdraw.filled_circle(
                self.window,
                asteroid_pos_2d[0],
                asteroid_pos_2d[1],
                size,
                (255, 215, 0),
            )

            # Add resource glow for high-value asteroids
            if resource_ratio > 0.5:
                pulse = math.sin(self.env.steps_count * 0.15) * 0.3 + 0.7
                glow_alpha = int(60 * pulse)
                for j in range(3):
                    alpha_val = max(0, glow_alpha - j * 20)
                    if alpha_val > 0:
                        glow_surface = pygame.Surface((size * 2 + j * 4 + 10, size * 2 + j * 4 + 10), pygame.SRCALPHA)
                        gfxdraw.filled_circle(glow_surface, size + j * 2 + 5, size + j * 2 + 5, size + j * 2, (255, 255, 100, alpha_val))
                        self.window.blit(glow_surface, (asteroid_pos_2d[0] - size - j * 2 - 5, asteroid_pos_2d[1] - size - j * 2 - 5), special_flags=pygame.BLEND_ADD)

            # Enhanced health bar
            health_ratio = resource_ratio
            health_width = int(28 * health_ratio)
            health_x = asteroid_pos_2d[0] - 14
            health_y = asteroid_pos_2d[1] - size - 12

            # Background
            pygame.draw.rect(self.window, (40, 40, 40), (health_x, health_y, 28, 6))
            # Health bar with fixed color scheme
            if health_ratio > 0.6:
                health_color = (0, 255, 0)
            elif health_ratio > 0.3:
                health_color = (255, 255, 0)
            else:
                health_color = (255, 100, 0)
            pygame.draw.rect(self.window, health_color, (health_x, health_y, health_width, 6))

            # Show resource value as text for larger asteroids
            if size > 12:
                resource_text = f"{self.env.asteroid_resources[i]:.0f}"
                text_surface = pygame.font.SysFont("Arial", 12).render(resource_text, True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=(asteroid_pos_2d[0], asteroid_pos_2d[1] + size + 18))
                self.window.blit(text_surface, text_rect)

        # Draw obstacles with enhanced pulsing effect
        for pos in self.env.obstacle_positions:
            obstacle_pos_2d = to_screen(pos)
            pulse = int(8 + 3 * math.sin(self.env.steps_count * 0.1))
            gfxdraw.filled_circle(
                self.window, obstacle_pos_2d[0], obstacle_pos_2d[1], pulse, (220, 50, 50)
            )
            # Enhanced danger glow
            for i in range(3):
                alpha_val = max(0, 100 - i * 30)
                if alpha_val > 0:
                    glow_surface = pygame.Surface((pulse * 2 + i * 4 + 20, pulse * 2 + i * 4 + 20), pygame.SRCALPHA)
                    gfxdraw.filled_circle(glow_surface, pulse + i * 2 + 10, pulse + i * 2 + 10, pulse + 2 + i * 2, (255, 100, 100, alpha_val))
                    self.window.blit(glow_surface, (obstacle_pos_2d[0] - pulse - i * 2 - 10, obstacle_pos_2d[1] - pulse - i * 2 - 10), special_flags=pygame.BLEND_ADD)

        # Draw agent with FIXED GREEN color
        agent_pos_2d = to_screen(self.env.agent_position)

        # Always green agent
        agent_color = (50, 255, 50)
        
        # Add energy warning halo if low energy
        if self.env.agent_energy < 30:
            warning_intensity = int(100 * (1 - self.env.agent_energy / 30.0))
            warning_pulse = math.sin(self.env.steps_count * 0.3) * 0.5 + 0.5
            for i in range(3):
                alpha_val = max(0, int(warning_intensity * warning_pulse) - i * 20)
                if alpha_val > 0:
                    warning_surface = pygame.Surface((28 + i * 6 + 10, 28 + i * 6 + 10), pygame.SRCALPHA)
                    gfxdraw.filled_circle(warning_surface, 14 + i * 3 + 5, 14 + i * 3 + 5, 14 + i * 3, (255, 0, 0, alpha_val))
                    self.window.blit(warning_surface, (agent_pos_2d[0] - 14 - i * 3 - 5, agent_pos_2d[1] - 14 - i * 3 - 5), special_flags=pygame.BLEND_ADD)

        gfxdraw.filled_circle(
            self.window,
            agent_pos_2d[0],
            agent_pos_2d[1],
            15,
            agent_color,
        )

        # Draw white outline for visibility
        gfxdraw.aacircle(
            self.window,
            agent_pos_2d[0],
            agent_pos_2d[1],
            15,
            (255, 255, 255),
        )

        # Draw enhanced animated mining beam
        if hasattr(self.env, "mining_asteroid_id"):
            asteroid_pos_2d = to_screen(
                self.env.asteroid_positions[self.env.mining_asteroid_id]
            )
            
            # Enhanced pulsing mining beam
            beam_length = math.sqrt((asteroid_pos_2d[0] - agent_pos_2d[0])**2 + (asteroid_pos_2d[1] - agent_pos_2d[1])**2)
            num_segments = int(beam_length / 6)
            beam_intensity = math.sin(self.env.steps_count * 0.4) * 0.3 + 0.7
            
            for i in range(num_segments):
                t1 = (i + self.env.mining_beam_offset % 1) / num_segments
                t2 = (i + 0.4 + self.env.mining_beam_offset % 1) / num_segments
                
                if t1 <= 1.0 and t2 <= 1.0:
                    x1 = int(agent_pos_2d[0] + t1 * (asteroid_pos_2d[0] - agent_pos_2d[0]))
                    y1 = int(agent_pos_2d[1] + t1 * (asteroid_pos_2d[1] - agent_pos_2d[1]))
                    x2 = int(agent_pos_2d[0] + t2 * (asteroid_pos_2d[0] - agent_pos_2d[0]))
                    y2 = int(agent_pos_2d[1] + t2 * (asteroid_pos_2d[1] - agent_pos_2d[1]))
                    
                    # Enhanced beam colors with intensity variation
                    if i % 2 == 0:
                        color = (255, int(255 * beam_intensity), 0)
                    else:
                        color = (255, int(200 * beam_intensity), 50)
                    pygame.draw.line(self.window, color, (x1, y1), (x2, y2), 5)

        # Draw inventory indicator
        if self.env.agent_inventory > 0:
            inventory_size = int(4 + self.env.agent_inventory / 8)
            gfxdraw.filled_circle(
                self.window,
                agent_pos_2d[0],
                agent_pos_2d[1],
                inventory_size,
                (255, 215, 0),
            )

        # Enhanced energy bar
        energy_ratio = self.env.agent_energy / 150.0
        energy_width = int(70 * energy_ratio)
        bar_x = agent_pos_2d[0] - 35
        bar_y = agent_pos_2d[1] - 35
        
        # Background with border
        pygame.draw.rect(self.window, (20, 20, 20), (bar_x - 2, bar_y - 2, 74, 12))
        pygame.draw.rect(self.window, (60, 60, 60), (bar_x, bar_y, 70, 8))
        
        # Energy bar with enhanced colors
        if energy_ratio > 0.6:
            energy_color = (0, 255, 0)
        elif energy_ratio > 0.3:
            energy_color = (255, 255, 0)
        else:
            energy_color = (255, 50, 50)
        pygame.draw.rect(self.window, energy_color, (bar_x, bar_y, energy_width, 8))

        # Draw observation and mining ranges (affected by zoom)
        obs_radius_px = int(self.env.observation_radius * 14.0 * self.env.zoom_level)
        mining_radius_px = int(self.env.mining_range * 14.0 * self.env.zoom_level)
        
        gfxdraw.aacircle(
            self.window,
            agent_pos_2d[0],
            agent_pos_2d[1],
            obs_radius_px,
            (100, 150, 255),
        )
        gfxdraw.aacircle(
            self.window,
            agent_pos_2d[0],
            agent_pos_2d[1],
            mining_radius_px,
            (255, 100, 100),
        )

        # Draw delivery particles
        for particle in self.env.delivery_particles:
            progress = particle["progress"]
            current_pos = (
                particle["start_pos"] + progress * (particle["target_pos"] - particle["start_pos"])
            )
            particle_pos_2d = to_screen(current_pos)
            
            alpha = int(255 * (1 - progress * 0.7))
            if alpha > 0:
                particle_surface = pygame.Surface((14, 14), pygame.SRCALPHA)
                glow_color = (255, 255, 0, alpha)
                gfxdraw.filled_circle(particle_surface, 7, 7, 5, glow_color)
                gfxdraw.filled_circle(particle_surface, 7, 7, 3, (255, 255, 255, alpha))
                self.window.blit(particle_surface, (particle_pos_2d[0] - 7, particle_pos_2d[1] - 7))

        # Draw score popups with enhanced styling
        for popup in self.env.score_popups:
            popup_pos_2d = to_screen(popup["pos"])
            alpha = max(0, min(255, popup["alpha"]))
            if alpha > 0:
                font_size = 18 if popup["alpha"] > 200 else 16
                popup_font = pygame.font.SysFont("Arial", font_size, bold=True)
                text_surface = popup_font.render(popup["text"], True, popup["color"])
                text_surface.set_alpha(alpha)
                
                # Add shadow effect
                shadow_surface = popup_font.render(popup["text"], True, (0, 0, 0))
                shadow_surface.set_alpha(alpha // 2)
                self.window.blit(shadow_surface, (popup_pos_2d[0] - 18, popup_pos_2d[1] - 8))
                self.window.blit(text_surface, (popup_pos_2d[0] - 20, popup_pos_2d[1] - 10))

        # OBSERVATION RANGE DIMMING EFFECT
        overlay = pygame.Surface((1920, 1200), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))  # 200/255 darkness outside observation circle
        
        # Create a "visible" hole for observation range
        pygame.draw.circle(overlay, (0, 0, 0, 0), agent_pos_2d, obs_radius_px)
        
        # Apply the dimming overlay
        self.window.blit(overlay, (0, 0))

        # Draw collision flash overlay (after dimming so it's always visible)
        if self.env.collision_flash_timer > 0:
            flash_alpha = int(160 * (self.env.collision_flash_timer / 0.3))  # Less intense for ultra-wide
            flash_surface = pygame.Surface((1920, 1200), pygame.SRCALPHA)
            flash_surface.fill((255, 0, 0, flash_alpha))
            self.window.blit(flash_surface, (0, 0))


        # Draw game UI if not in game over state
        if not self.env.game_over_state["active"]:
            self._draw_game_ui(agent_pos_2d)
        else:
            self._draw_game_over_screen()

        # Update display / timing
        if self.env.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.env.metadata["render_fps"])

        if self.env.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def _draw_starfield(self) -> None:
        """Draw perfect starfield for cosmic depth."""
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            return
            
        for layer in self.starfield_layers:
            for star in layer:
                if not (0 <= star["x"] <= 1920 and 0 <= star["y"] <= 1080):
                    continue
                    
                x, y = int(star["x"]), int(star["y"])
                size = star["size"]
                brightness = star["brightness"]
                
                # Perfect cosmic star colors
                if star["color_type"] == "blue":
                    color = (brightness//5, brightness//2, brightness)
                elif star["color_type"] == "yellow":
                    color = (brightness, brightness//1.4, brightness//5)
                else:  # white
                    color = (brightness, brightness, brightness)
                
                # Clean star rendering
                if size == 1:
                    gfxdraw.pixel(self.window, x, y, color)
                else:
                    gfxdraw.filled_circle(self.window, x, y, size-1, color)
                    if size == 3:  # Only large stars get glow
                        glow = (*[c//4 for c in color], 35)
                        gfxdraw.filled_circle(self.window, x, y, size, glow)

    # Nebula drawing intentionally removed (scope simplified to starfield)

    # Galaxy drawing intentionally removed (scope simplified to starfield)

    # Space dust drawing intentionally removed (scope simplified to starfield)

    # Aurora drawing intentionally removed (scope simplified to starfield)





    def _draw_game_ui(self, agent_pos_2d) -> None:
        """Draw the main game UI elements with adaptive layout and icons."""
        try:
            import pygame
            import numpy as np
            from pygame import gfxdraw
            import math
        except ImportError:
            return

        # Enhanced status panel with adaptive layout
        cumulative_mining = getattr(self.env, "cumulative_mining_amount", 0.0)

        # Agent state color (avoid RL-term 'Explore/Exploring' in UI)
        # Title should be static and simple for clarity
        state_color = (200, 200, 255)
        if hasattr(self.env, "mining_asteroid_id") and self.env.mining_asteroid_id is not None:
            state_color = (255, 255, 100)
        elif self.env.agent_inventory > 0:
            state_color = (255, 255, 0)

        # Create adaptive status panel with a static title
        self._draw_adaptive_status_panel("STATUS", state_color, cumulative_mining)
        
        # Create adaptive legend with icons
        self._draw_adaptive_legend()

    def _draw_adaptive_status_panel(self, state_text, state_color, cumulative_mining) -> None:
        """Draw an adaptive status panel with icons stacked vertically on the left."""
        try:
            import pygame
            import numpy as np
            from pygame import gfxdraw
            import math
        except ImportError:
            return

        # Compute metrics
        fitness = self.env.compute_fitness_score()
        speed = float(np.linalg.norm(self.env.agent_velocity))
        vx, vy = float(self.env.agent_velocity[0]), float(self.env.agent_velocity[1])
        # Heading in degrees; if nearly stationary, show N/A later
        heading = (math.degrees(math.atan2(vy, vx)) % 360) if speed > 1e-4 else None
        dist_to_mothership = float(np.linalg.norm(self.env.agent_position - self.env.mothership_pos))

        # Simple state label without using 'Explore/Exploring'
        if hasattr(self.env, "mining_asteroid_id") and self.env.mining_asteroid_id is not None:
            state_label = f"Mining A{self.env.mining_asteroid_id}"
            mining_status = f"Active (A{self.env.mining_asteroid_id})"
        elif self.env.agent_inventory > 0:
            state_label = f"Carrying {self.env.agent_inventory:.0f}"
            mining_status = "Not mining"
        else:
            state_label = "Searching"
            mining_status = "Not mining"

        # Compact values
        energy_pct = int(self.env.agent_energy / 150.0 * 100)
        inv = self.env.agent_inventory
        inv_max = getattr(self.env, "max_inventory", 100)

        # Build items (ordered) with clear, concise labels
        status_items = [
            {"icon": "state", "value": f"State: {state_label}", "warning": False, "color": (220, 220, 255)},
            {"icon": "step", "value": f"Step Count: {self.env.steps_count} of {self.env.max_episode_steps}", "warning": False, "color": (200, 200, 255)},
            {"icon": "energy", "value": f"Energy Level: {int(self.env.agent_energy)}/150 ({energy_pct}%)", "warning": energy_pct < 30, "color": (255, 100, 100) if energy_pct < 30 else (100, 255, 100)},
            {"icon": "inventory", "value": f"Inventory: {inv:.1f} of {inv_max} Capacity", "warning": False, "color": (255, 255, 100) if inv > 0 else (180, 180, 180)},
            {"icon": "mining", "value": f"Mining: {mining_status}", "warning": False, "color": (255, 255, 100)},
            {"icon": "delivery", "value": f"Deliveries: {getattr(self.env, 'delivery_count', 0)}", "warning": False, "color": (0, 255, 0)},
            {"icon": "score", "value": f"Fitness Score: {fitness:.1f}", "warning": False, "color": (255, 215, 100)},
            {"icon": "velocity", "value": f"Speed: {speed:.2f}", "warning": False, "color": (180, 180, 255)},
            {"icon": "velocity", "value": f"Speed X: {vx:.2f}", "warning": False, "color": (180, 180, 255)},
            {"icon": "velocity", "value": f"Speed Y: {vy:.2f}", "warning": False, "color": (180, 180, 255)},
            {"icon": "angle", "value": f"Heading (deg): {heading:.0f}" if heading is not None else "Heading (deg): N/A", "warning": False, "color": (180, 180, 255)},
            {"icon": "mothership", "value": f"Distance to Mothership: {dist_to_mothership:.1f}", "warning": False, "color": (30, 120, 200)},
            {"icon": "collision", "value": f"Collisions: {self.env.collision_count}", "warning": self.env.collision_count > 0, "color": (255, 100, 100) if self.env.collision_count > 0 else (200, 200, 200)},
            {"icon": "action", "value": f"Action: ({self.env.last_action[0]:.2f}, {self.env.last_action[1]:.2f}, {'Mining' if self.env.last_action[2]>0.5 else 'Not Mining'})", "warning": False, "color": (200, 200, 200)},
        ]

        # Layout: vertical stack on left, less compact for clarity
        item_width = 320
        item_height = 38
        padding = 14
        panel_width = item_width + padding * 2

        # Calculate height with score key section
        title_height = 30
        items_height = len(status_items) * item_height
        score_gap = 14
        score_row_h = 26
        panel_height = title_height + items_height + score_gap + score_row_h + 30

        # Create main status panel
        status_bg = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        status_bg.fill((0, 0, 0, 200))
        pygame.draw.rect(status_bg, (60, 60, 80), (0, 0, panel_width, panel_height), 2)

        # Title (larger for clarity)
        title_font = pygame.font.SysFont("Arial", 18, bold=True)
        title_surface = title_font.render(state_text, True, state_color)
        title_rect = title_surface.get_rect(center=(panel_width // 2, 22))
        status_bg.blit(title_surface, title_rect)

        # Draw status items vertically
        for i, item in enumerate(status_items):
            y = 44 + i * item_height
            x = padding

            # Icon
            icon_x = x
            icon_y = y + 8
            self._draw_status_icon(status_bg, item["icon"], icon_x, icon_y, item["warning"])

            # Text
            value_font = pygame.font.SysFont("Arial", 16, bold=item["warning"])
            value_surface = value_font.render(item["value"], True, item["color"])
            status_bg.blit(value_surface, (icon_x + 38, y + 10))

        # Blit to left edge (15 px margin)
        self.window.blit(status_bg, (15, 15))

    def _draw_status_icon(self, surface, icon_type, x, y, warning=False) -> None:
        """Draw a small status icon."""
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            return

        size = 20
        warning_color = (255, 100, 100) if warning else None

        if icon_type == "energy":
            # Battery icon
            color = warning_color or (100, 255, 100)
            pygame.draw.rect(surface, color, (x, y+5, 16, 10), 2)
            pygame.draw.rect(surface, color, (x+16, y+7, 3, 6))
            # Fill based on energy
            fill_width = int(14 * (self.env.agent_energy / 150.0))
            pygame.draw.rect(surface, color, (x+1, y+6, fill_width, 8))

        elif icon_type == "inventory":
            # Package/box icon
            color = warning_color or (255, 255, 100)
            pygame.draw.rect(surface, color, (x+2, y+3, 14, 14), 2)
            pygame.draw.line(surface, color, (x+2, y+8), (x+16, y+8), 2)
            pygame.draw.line(surface, color, (x+9, y+3), (x+9, y+17), 2)

        elif icon_type == "mining":
            # Pickaxe icon
            color = warning_color or (100, 255, 100)
            pygame.draw.line(surface, color, (x+3, y+15), (x+15, y+3), 3)
            pygame.draw.rect(surface, color, (x+12, y+1, 6, 4))

        elif icon_type == "collision":
            # Warning/explosion icon
            color = warning_color or (200, 200, 200)
            gfxdraw.filled_circle(surface, x+10, y+10, 8, color)
            font = pygame.font.SysFont("Arial", 12, bold=True)
            text = font.render("!", True, (0, 0, 0))
            surface.blit(text, (x+7, y+4))

        elif icon_type == "step":
            # Clock icon
            color = warning_color or (200, 200, 255)
            gfxdraw.aacircle(surface, x+10, y+10, 8, color)
            pygame.draw.line(surface, color, (x+10, y+10), (x+10, y+5), 2)
            pygame.draw.line(surface, color, (x+10, y+10), (x+14, y+10), 2)

        elif icon_type == "asteroid":
            # Asteroid icon (rough circle)
            color = warning_color or (255, 215, 100)
            points = [(x+10, y+2), (x+16, y+6), (x+15, y+14), (x+8, y+17), (x+3, y+12), (x+4, y+5)]
            pygame.draw.polygon(surface, color, points, 2)

        elif icon_type == "mothership":
            # Mothership icon
            color = warning_color or (30, 120, 200)
            gfxdraw.filled_circle(surface, x+10, y+10, 8, color)

        elif icon_type == "delivery":
            # Delivery icon
            color = warning_color or (0, 255, 0)
            pygame.draw.rect(surface, color, (x+3, y+5, 14, 10), 2)
            pygame.draw.line(surface, color, (x+3, y+10), (x+17, y+10), 2)

        elif icon_type == "score":
            # Score icon
            color = warning_color or (255, 215, 100)
            pygame.draw.polygon(surface, color, [(x+10, y+2), (x+18, y+18), (x+2, y+18)], 2)

        elif icon_type == "velocity":
            # Velocity icon (arrow)
            color = warning_color or (180, 180, 255)
            pygame.draw.line(surface, color, (x+5, y+10), (x+15, y+10), 2)
            pygame.draw.line(surface, color, (x+12, y+7), (x+15, y+10), 2)
            pygame.draw.line(surface, color, (x+12, y+13), (x+15, y+10), 2)

        elif icon_type == "angle":
            # Heading icon (compass)
            color = warning_color or (180, 180, 255)
            gfxdraw.aacircle(surface, x+10, y+10, 8, color)
            pygame.draw.line(surface, color, (x+10, y+2), (x+10, y+10), 2)

        elif icon_type == "action":
            # Action icon (controller)
            color = warning_color or (200, 200, 200)
            pygame.draw.rect(surface, color, (x+2, y+6, 16, 8), 2)
            pygame.draw.rect(surface, color, (x+12, y+2, 4, 4), 2)

        elif icon_type == "state":
            # State icon (agent dot)
            color = warning_color or (200, 200, 255)
            gfxdraw.filled_circle(surface, x+10, y+10, 6, color)
            gfxdraw.aacircle(surface, x+10, y+10, 6, (255, 255, 255))


    def _draw_adaptive_legend(self) -> None:
        """Draw an adaptive legend — vertical, right-anchored to avoid overflow."""
        try:
            import pygame
            from pygame import gfxdraw
            import math
        except ImportError:
            return

        legend_items = [
            {"icon": "agent", "text": "Mining Agent", "color": (50, 255, 50)},
            {"icon": "mothership", "text": "Mothership Base", "color": (30, 120, 200)},
            {"icon": "asteroid", "text": "Resource Asteroids", "color": (255, 215, 0)},
            {"icon": "obstacle", "text": "Hazardous Obstacles", "color": (220, 50, 50)},
            {"icon": "obs_range", "text": "Observation Range", "color": (100, 150, 255)},
            {"icon": "mine_range", "text": "Mining Range", "color": (255, 100, 100)}
        ]

        # Score popup color key (integrated into the legend panel)
        score_key_items = [
            {"text": "+X: Resources Delivered to Base", "color": (0, 255, 0)},
            {"text": "+X: Mining Resources from Asteroid", "color": (255, 255, 0)},
            {"text": "+X: Energy Recharge at Base", "color": (100, 150, 255)}
        ]

        # Layout metrics
        item_height = 38
        item_width = 300
        padding = 16
        score_header_height = 24
        score_item_height = 30
        score_section_gap = 12

        panel_width = item_width + padding * 2
        panel_height = (
            len(legend_items) * item_height + 48
            + score_section_gap + score_header_height
            + len(score_key_items) * score_item_height
        )

        # Create legend background
        legend_bg = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        legend_bg.fill((0, 0, 0, 200))
        pygame.draw.rect(legend_bg, (60, 60, 80), (0, 0, panel_width, panel_height), 2)

        # Title
        title_font = pygame.font.SysFont("Arial", 18, bold=True)
        title_surface = title_font.render("LEGEND", True, (255, 255, 150))
        title_rect = title_surface.get_rect(center=(panel_width // 2, 22))
        legend_bg.blit(title_surface, title_rect)

        # Draw standard legend items (icons)
        for i, item in enumerate(legend_items):
            y = 44 + i * item_height
            x = padding

            # icon at left of row
            self._draw_legend_icon(legend_bg, item["icon"], x + 4, y + 8, item["color"])

            # text
            text_font = pygame.font.SysFont("Arial", 16)
            text_surface = text_font.render(item["text"], True, (220, 220, 220))
            legend_bg.blit(text_surface, (x + 38, y + 10))

        # Score section header (without separator line)
        section_top = 44 + len(legend_items) * item_height + score_section_gap

        score_title_font = pygame.font.SysFont("Arial", 16, bold=True)
        score_title_surface = score_title_font.render("SCORE KEY", True, (200, 220, 255))
        score_title_rect = score_title_surface.get_rect(center=(panel_width // 2, section_top))
        legend_bg.blit(score_title_surface, score_title_rect)

        # Draw score color key items
        start_y = section_top + score_header_height
        for i, item in enumerate(score_key_items):
            y = start_y + i * score_item_height
            x = padding

            # Color chip
            chip_rect = pygame.Rect(x + 4, y + 7, 24, 16)
            pygame.draw.rect(legend_bg, item["color"], chip_rect)
            pygame.draw.rect(legend_bg, (40, 40, 40), chip_rect, 1)

            # Description text
            text_font = pygame.font.SysFont("Arial", 14)
            text_surface = text_font.render(item["text"], True, (220, 220, 220))
            legend_bg.blit(text_surface, (x + 34, y + 6))

        # Position legend bottom-right with safe margins for 1080p board
        margin = 20
        legend_x = 1920 - panel_width - margin
        legend_y = 1080 - panel_height - margin
        self.window.blit(legend_bg, (legend_x, legend_y))

    def _draw_legend_icon(self, surface, icon_type, x, y, color) -> None:
        """Draw legend icons that match the actual game elements."""
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            return

        if icon_type == "agent":
            # Agent circle
            gfxdraw.filled_circle(surface, x+10, y+10, 8, color)
            gfxdraw.aacircle(surface, x+10, y+10, 8, (255, 255, 255))

        elif icon_type == "mothership":
            # Mothership with glow
            gfxdraw.filled_circle(surface, x+10, y+10, 8, color)
            gfxdraw.aacircle(surface, x+10, y+10, 10, color)

        elif icon_type == "asteroid":
            # Asteroid shape
            points = [(x+10, y+2), (x+16, y+6), (x+15, y+14), (x+8, y+17), (x+3, y+12), (x+4, y+5)]
            gfxdraw.filled_polygon(surface, points, color)

        elif icon_type == "obstacle":
            # Pulsing obstacle
            gfxdraw.filled_circle(surface, x+10, y+10, 7, color)
            gfxdraw.aacircle(surface, x+10, y+10, 9, (255, 100, 100))

        elif icon_type == "depleted":
            # Gray circle with X
            gfxdraw.filled_circle(surface, x+10, y+10, 6, color)
            pygame.draw.line(surface, (200, 200, 200), (x+6, y+6), (x+14, y+14), 2)
            pygame.draw.line(surface, (200, 200, 200), (x+14, y+6), (x+6, y+14), 2)

        elif icon_type == "obs_range":
            # Observation range circle
            gfxdraw.aacircle(surface, x+10, y+10, 8, color)

        elif icon_type == "mine_range":
            # Mining range circle
            gfxdraw.aacircle(surface, x+10, y+10, 6, color)

        elif icon_type == "safe_zone":
            # Safe zone aura
            for i in range(3):
                alpha = max(0, 100 - i * 30)
                if alpha > 0:
                    # Create a temporary surface for the aura effect
                    aura_surface = pygame.Surface((20 + i * 4, 20 + i * 4), pygame.SRCALPHA)
                    gfxdraw.filled_circle(aura_surface, 10 + i * 2, 10 + i * 2, 8 + i * 2, (*color[:3], alpha))
                    surface.blit(aura_surface, (x - i * 2, y - i * 2), special_flags=pygame.BLEND_ADD)

        elif icon_type == "energy_beam":
            # Mining beam
            pygame.draw.line(surface, color, (x+3, y+15), (x+17, y+5), 3)

        elif icon_type == "particles":
            # Delivery particles
            for i, pos in enumerate([(x+5, y+8), (x+10, y+6), (x+15, y+10)]):
                gfxdraw.filled_circle(surface, pos[0], pos[1], 2, color)

        elif icon_type == "trail":
            # Agent trail
            for i, alpha in enumerate([255, 180, 120]):
                gfxdraw.filled_circle(surface, x+4+i*4, y+10, 2, (*color[:3], alpha))

        elif icon_type == "dim_area":
            # Dimmed area
            pygame.draw.rect(surface, color, (x+3, y+3, 14, 14))
            gfxdraw.filled_circle(surface, x+10, y+10, 5, (0, 0, 0, 0))


    def _draw_game_over_screen(self) -> None:
        """Draw the game over/success screen with final statistics."""
        try:
            import pygame
        except ImportError:
            return

        # Update fade alpha
        if self.env.game_over_state["fade_alpha"] < 255:
            self.env.game_over_state["fade_alpha"] += 3

        fade_alpha = min(255, self.env.game_over_state["fade_alpha"])
        
        # Create fade overlay for ultra-wide screen
        fade_surface = pygame.Surface((1920, 1200), pygame.SRCALPHA)
        fade_surface.fill((0, 0, 0, fade_alpha))
        self.window.blit(fade_surface, (0, 0))

        # Only show text once fade is substantial
        if fade_alpha > 100:
            stats = self.env.game_over_state["final_stats"]
            success = self.env.game_over_state["success"]
            
            # Title for ultra-wide screen
            title_font = pygame.font.SysFont("Arial", 72, bold=True)
            title_text = "🎉 MISSION SUCCESS! 🎉" if success else "💥 MISSION FAILED 💥"
            title_color = (0, 255, 0) if success else (255, 100, 100)
            title_surface = title_font.render(title_text, True, title_color)
            title_rect = title_surface.get_rect(center=(960, 260))
            self.window.blit(title_surface, title_rect)

            # Final statistics
            stats_font = pygame.font.SysFont("Arial", 26)
            stats_text = [
                "",
                f"📊 FINAL STATISTICS",
                "",
                f"🌕 Total Resources Mined: {stats['total_resources_mined']:.1f}",
                f"🚀 Resources Delivered: {stats['resources_delivered']:.1f}",
                f"📦 Final Inventory: {stats['current_inventory']:.1f}",
                f"💥 Collisions: {stats['collisions']}",
                f"⏱️ Steps Taken: {stats['steps_taken']}",
                f"⚡ Final Energy: {stats['final_energy']:.1f}/150",
                f"💀 Asteroids Depleted: {stats['asteroids_depleted']}/{stats['total_asteroids']}",
                f"🏆 Efficiency Score: {stats['efficiency_score']:.0f}",
                "",
                "Press R to restart or ESC to quit"
            ]

            y_offset = 400
            for line in stats_text:
                if line == "":
                    y_offset += 26
                    continue
                    
                color = (255, 255, 100) if "STATISTICS" in line else (255, 255, 255)
                if "restart" in line:
                    color = (200, 200, 255)
                
                text_surface = stats_font.render(line, True, color)
                text_rect = text_surface.get_rect(center=(960, y_offset))
                self.window.blit(text_surface, text_rect)
                y_offset += 45

    def update_zoom(self) -> None:
        """Update zoom system for cosmic immersion."""
        zoom_diff = self.env.target_zoom - self.env.zoom_level
        self.env.zoom_speed = 0.025
        self.env.zoom_level += zoom_diff * self.env.zoom_speed
        
        # Perfect zoom logic for cosmic experience
        if hasattr(self.env, 'agent_energy') and self.env.agent_energy < 20:
            self.env.target_zoom = 1.25  # Focus when energy critical
        elif hasattr(self.env, 'collision_flash_timer') and self.env.collision_flash_timer > 0:
            self.env.target_zoom = 0.85  # Pull back for drama
        elif len([a for a in self.env.asteroid_resources if a > 0.1]) <= 2:
            self.env.target_zoom = 0.9   # Overview when few asteroids
        elif hasattr(self.env, 'mining_asteroid_id') and self.env.mining_asteroid_id is not None:
            self.env.target_zoom = 1.1   # Subtle mining focus
        else:
            self.env.target_zoom = 1.0   # Perfect cosmic view
        
        # Perfect zoom bounds for cosmic viewing
        self.env.zoom_level = max(0.8, min(1.3, self.env.zoom_level))



    def spawn_delivery_particles(self, start_pos: np.ndarray, target_pos: np.ndarray) -> None:
        """Spawn glowing particles for resource delivery animation."""
        for _ in range(10):
            particle = {
                "start_pos": start_pos.copy(),
                "target_pos": target_pos.copy(),
                "progress": 0.0
            }
            self.env.delivery_particles.append(particle)

    def add_score_popup(self, text: str, pos: np.ndarray, color: tuple) -> None:
        """Add a floating score popup."""
        popup = {
            "text": text,
            "pos": pos.copy(),
            "alpha": 255,
            "color": color
        }
        self.env.score_popups.append(popup)

    def update_animations(self) -> None:
        """Update all animation states."""
        # Update delivery particles
        for particle in self.env.delivery_particles:
            particle["progress"] += 0.05
            if particle["progress"] >= 1.0:
                particle["progress"] = 1.0
        # Remove completed particles
        self.env.delivery_particles = [p for p in self.env.delivery_particles if p["progress"] < 1.0]

        # Update score popups
        for popup in self.env.score_popups:
            popup["pos"][1] -= 0.3  # Move upward
            popup["alpha"] -= 5
        # Remove faded popups
        self.env.score_popups = [p for p in self.env.score_popups if p["alpha"] > 0]

        # Update collision effects
        if self.env.collision_flash_timer > 0:
            self.env.collision_flash_timer -= self.env.dt
        if self.env.screen_shake_timer > 0:
            self.env.screen_shake_timer -= self.env.dt

        # Update mining beam animation
        self.env.mining_beam_offset += 0.2


    def trigger_game_over(self, success: bool) -> None:
        """Trigger game over screen with final statistics."""
        cumulative_mining = getattr(self.env, "cumulative_mining_amount", 0.0)
        
        self.env.game_over_state = {
            "active": True,
            "fade_alpha": 0,
            "success": success,
            "final_stats": {
                "total_resources_mined": cumulative_mining,
                "resources_delivered": cumulative_mining - self.env.agent_inventory,
                "current_inventory": self.env.agent_inventory,
                "collisions": self.env.collision_count,
                "steps_taken": self.env.steps_count,
                "final_energy": self.env.agent_energy,
                "asteroids_depleted": len(self.env.asteroid_positions) - np.sum(self.env.asteroid_resources >= 0.1),
                "total_asteroids": len(self.env.asteroid_positions),
                "efficiency_score": self.env.compute_fitness_score()
            }
        }

    def close(self) -> None:
        """Close the rendering window."""
        # Defer import to avoid hard dependency at import-time
        try:
            import pygame
        except ImportError:
            pygame = None  # type: ignore
        if self.window is not None:
            if pygame and self.env.render_mode == "human":
                pygame.display.quit()
                pygame.quit()
            self.window = None
            self.clock = None
