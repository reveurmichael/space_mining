from typing import Any, Optional

import numpy as np


class Renderer:
    """Handles rendering for the SpaceMining environment."""

    def __init__(self, env: Any) -> None:
        self.env: Any = env
        self.window: Optional[Any] = None
        self.clock: Optional[Any] = None
        self.font: Optional[Any] = None

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
                self.window = pygame.display.set_mode((1440, 1080))  # 4:3 HD ratio for cinematic feel
                pygame.display.set_caption("🚀 Space Mining Universe - Cosmic Explorer")
            else:
                # Off-screen surface for rgb_array mode
                self.window = pygame.Surface((1440, 1080))
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

        # Deep space background
        self.window.fill((5, 5, 15))

        # Draw enhanced cosmic background
        self._draw_cosmic_background()

        # Helper function to convert 2D coordinates to screen coordinates with zoom
        def to_screen(pos, scale=12.0):  # Increased scale for larger screen
            x, y = pos
            zoom_scale = scale * self.env.zoom_level
            screen_x = int(720 + (x - 40) * zoom_scale + shake_offset[0])  # Center at 1440/2
            screen_y = int(540 + (y - 40) * zoom_scale + shake_offset[1])  # Center at 1080/2
            return screen_x, screen_y

        # Draw agent trail first (behind everything)
        for trail_point in self.env.agent_trail:
            trail_pos_2d = to_screen(trail_point["pos"])
            alpha = max(0, min(255, trail_point["alpha"]))
            if alpha > 0:
                trail_surface = pygame.Surface((10, 10), pygame.SRCALPHA)
                trail_color = (50, 255, 50, alpha // 2)
                gfxdraw.filled_circle(trail_surface, 5, 5, 3, trail_color)
                self.window.blit(trail_surface, (trail_pos_2d[0] - 5, trail_pos_2d[1] - 5))

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
            gfxdraw.aacircle(
                self.window, 
                mothership_pos_2d[0], 
                mothership_pos_2d[1], 
                20 + i * 4, 
                (30, 120, 200, alpha)
            )

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
                    gfxdraw.aacircle(
                        self.window,
                        asteroid_pos_2d[0],
                        asteroid_pos_2d[1],
                        size + j * 2,
                        (255, 255, 100, glow_alpha - j * 20)
                    )

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
                gfxdraw.aacircle(
                    self.window, obstacle_pos_2d[0], obstacle_pos_2d[1], 
                    pulse + 2 + i * 2, (255, 100, 100, 100 - i * 30)
                )

        # Draw agent with FIXED GREEN color
        agent_pos_2d = to_screen(self.env.agent_position)

        # Always green agent
        agent_color = (50, 255, 50)
        
        # Add energy warning halo if low energy
        if self.env.agent_energy < 30:
            warning_intensity = int(100 * (1 - self.env.agent_energy / 30.0))
            warning_pulse = math.sin(self.env.steps_count * 0.3) * 0.5 + 0.5
            for i in range(3):
                gfxdraw.aacircle(
                    self.window,
                    agent_pos_2d[0],
                    agent_pos_2d[1],
                    14 + i * 3,
                    (255, 0, 0, int(warning_intensity * warning_pulse) - i * 20)
                )

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
        obs_radius_px = int(self.env.observation_radius * 12.0 * self.env.zoom_level)
        mining_radius_px = int(self.env.mining_range * 12.0 * self.env.zoom_level)
        
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
        overlay = pygame.Surface((1440, 1080), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))  # Slightly less dimming for better cosmic view
        
        # Create a "visible" hole for observation range
        pygame.draw.circle(overlay, (0, 0, 0, 0), agent_pos_2d, obs_radius_px)
        
        # Apply the dimming overlay
        self.window.blit(overlay, (0, 0))

        # Draw collision flash overlay (after dimming so it's always visible)
        if self.env.collision_flash_timer > 0:
            flash_alpha = int(180 * (self.env.collision_flash_timer / 0.3))  # Slightly less intense
            flash_surface = pygame.Surface((1440, 1080), pygame.SRCALPHA)
            flash_surface.fill((255, 0, 0, flash_alpha))
            self.window.blit(flash_surface, (0, 0))

        # Draw floating event timeline
        self._draw_floating_timeline()

        # Draw combo display
        self._draw_combo_display()

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

    def _draw_cosmic_background(self) -> None:
        """Draw the enhanced cosmic background with nebulae, galaxies, and stars."""
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            return

        # Draw nebula clouds first (background layer)
        self._draw_nebulae()
        
        # Draw distant galaxies
        self._draw_distant_galaxies()
        
        # Draw space dust
        self._draw_space_dust()
        
        # Draw enhanced starfield with colors and twinkling
        self._draw_enhanced_starfield()

    def _draw_nebulae(self) -> None:
        """Draw colorful nebula clouds."""
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            return

        for nebula in self.env.nebula_clouds:
            x, y = int(nebula["x"]), int(nebula["y"])
            size = int(nebula["size"] * self.env.zoom_level)
            
            # Skip if completely off screen
            if x < -size or x > 1440 + size or y < -size or y > 1080 + size:
                continue
            
            # Create enhanced nebula effect with pulsing
            r, g, b, base_alpha = nebula["color"]
            
            # Add pulsing effect
            pulse = math.sin(self.env.cosmic_time * nebula.get("pulse_speed", 0.02) + nebula.get("pulse_offset", 0)) * 0.4 + 0.6
            
            # Enhanced multi-layered nebula
            for i in range(7):  # More layers for better effect
                layer_size = int(size * (1.0 - i * 0.12))
                if layer_size > 5:
                    alpha = int((base_alpha * pulse - i * 4) * (1.0 - i * 0.1))
                    alpha = max(3, min(alpha, 50))
                    
                    # Enhanced gradient with inner glow
                    color_variation = 1.0 - (i * 0.1)
                    inner_glow = 1.0 + (0.3 if i < 3 else 0)  # Brighter inner core
                    
                    final_color = (
                        int(min(255, r * color_variation * inner_glow)),
                        int(min(255, g * color_variation * inner_glow)), 
                        int(min(255, b * color_variation * inner_glow)),
                        alpha
                    )
                    
                    # Create nebula surface with transparency
                    nebula_surface = pygame.Surface((layer_size * 2, layer_size * 2), pygame.SRCALPHA)
                    gfxdraw.filled_circle(nebula_surface, layer_size, layer_size, layer_size, final_color)
                    
                    # Apply rotation if needed
                    if nebula["rotation"] != 0:
                        nebula_surface = pygame.transform.rotate(nebula_surface, math.degrees(nebula["rotation"]))
                    
                    # Blit to main surface
                    blit_x = x - nebula_surface.get_width() // 2
                    blit_y = y - nebula_surface.get_height() // 2
                    self.window.blit(nebula_surface, (blit_x, blit_y))

    def _draw_distant_galaxies(self) -> None:
        """Draw distant spiral galaxies."""
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            return

        for galaxy in self.env.distant_galaxies:
            x, y = int(galaxy["x"]), int(galaxy["y"])
            size = int(galaxy["size"] * self.env.zoom_level)
            
            # Skip if off screen
            if x < -size or x > 1440 + size or y < -size or y > 1080 + size:
                continue
            
            brightness = galaxy["brightness"]
            core_brightness = galaxy.get("core_brightness", brightness + 20)
            arm_thickness = galaxy.get("arm_thickness", 1.0)
            
            # Enhanced galaxy core with glow
            core_color = (core_brightness + 15, core_brightness + 10, core_brightness + 20)
            core_size = max(2, size // 6)
            gfxdraw.filled_circle(self.window, x, y, core_size, core_color)
            
            # Core glow
            if core_size > 2:
                glow_color = (core_brightness // 2, core_brightness // 2, core_brightness // 2 + 5)
                gfxdraw.aacircle(self.window, x, y, core_size + 1, glow_color)
            
            # Enhanced spiral arms
            arm_count = galaxy["spiral_arms"]
            rotation = galaxy["rotation"]
            
            for arm in range(arm_count):
                arm_angle = rotation + (arm * 2 * math.pi / arm_count)
                
                # Draw enhanced spiral arm
                for r in range(size // 8, size, max(1, size // 25)):
                    spiral_angle = arm_angle + r * 0.25  # Tighter spiral
                    
                    arm_x = x + int(r * math.cos(spiral_angle))
                    arm_y = y + int(r * math.sin(spiral_angle))
                    
                    if 0 <= arm_x <= 1440 and 0 <= arm_y <= 1080:
                        # Enhanced fade with distance
                        fade_factor = (1.0 - (r / size)) ** 1.5
                        arm_brightness = int(brightness * fade_factor)
                        
                        if arm_brightness > 3:
                            arm_color = (
                                arm_brightness, 
                                arm_brightness, 
                                min(255, arm_brightness + 8)
                            )
                            
                            # Variable thickness based on distance
                            if r < size // 2:
                                dot_size = max(1, int(2 * arm_thickness))
                                if dot_size > 1:
                                    gfxdraw.filled_circle(self.window, arm_x, arm_y, dot_size, arm_color)
                                else:
                                    self.window.set_at((arm_x, arm_y), arm_color)
                            else:
                                self.window.set_at((arm_x, arm_y), arm_color)

    def _draw_space_dust(self) -> None:
        """Draw fine cosmic dust particles."""
        try:
            import pygame
        except ImportError:
            return

        for dust in self.env.space_dust:
            x, y = int(dust["x"]), int(dust["y"])
            
            if 0 <= x <= 1440 and 0 <= y <= 1080:
                brightness = dust["brightness"]
                dust_type = dust.get("type", "fine")
                
                # Different rendering for different dust types
                if dust_type == "coarse":
                    # Larger, more visible dust
                    dust_color = (brightness + 5, brightness + 3, brightness + 8)
                    size = max(1, int(dust["size"] * self.env.zoom_level))
                    
                    if size > 1:
                        gfxdraw.filled_circle(self.window, x, y, size, dust_color)
                        # Add subtle glow for larger particles
                        if size > 2:
                            glow_color = (brightness // 2, brightness // 2, brightness // 2)
                            gfxdraw.aacircle(self.window, x, y, size + 1, glow_color)
                    else:
                        self.window.set_at((x, y), dust_color)
                else:
                    # Fine cosmic dust
                    dust_color = (brightness, brightness, brightness + 2)
                    size = max(1, int(dust["size"] * self.env.zoom_level))
                    
                    if size == 1:
                        self.window.set_at((x, y), dust_color)
                    else:
                        gfxdraw.filled_circle(self.window, x, y, size, dust_color)

    def _draw_enhanced_starfield(self) -> None:
        """Draw enhanced starfield with colors and twinkling."""
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            return

        for layer in self.env.starfield_layers:
            for star in layer:
                x, y = int(star["x"]), int(star["y"])
                
                if 0 <= x <= 1440 and 0 <= y <= 1080:
                    size = max(1, int(star["size"] * self.env.zoom_level))
                    base_brightness = star["brightness"]
                    
                    # Enhanced twinkling effect with variable speed
                    twinkle_speed = star.get("twinkle_speed", 1.0)
                    twinkle = math.sin(self.env.cosmic_time * twinkle_speed + star["twinkle_offset"]) * 0.4 + 0.6
                    brightness = int(base_brightness * twinkle)
                    brightness = max(15, min(255, brightness))
                    
                    # Color based on star type
                    color_type = star["color_type"]
                    if color_type == "blue":
                        color = (brightness // 2, brightness // 2, brightness)
                    elif color_type == "yellow":
                        color = (brightness, brightness, brightness // 2)
                    elif color_type == "red":
                        color = (brightness, brightness // 2, brightness // 2)
                    else:  # white
                        color = (brightness, brightness, brightness)
                    
                    if size == 1:
                        self.window.set_at((x, y), color)
                    else:
                        gfxdraw.filled_circle(self.window, x, y, size, color)
                        
                        # Add glow for larger stars
                        if size > 2:
                            glow_brightness = brightness // 3
                            glow_color = (glow_brightness, glow_brightness, glow_brightness)
                            gfxdraw.aacircle(self.window, x, y, size + 1, glow_color)

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
        
        # Agent state indicator
        state_text = "EXPLORING"
        state_color = (200, 200, 255)
        if hasattr(self.env, "mining_asteroid_id"):
            state_text = f"MINING A{self.env.mining_asteroid_id}"
            state_color = (255, 255, 100)
        elif self.env.agent_inventory > 0:
            state_text = f"CARRYING {self.env.agent_inventory:.0f}"
            state_color = (255, 255, 0)
        
        # Create adaptive status panel with icons
        self._draw_adaptive_status_panel(state_text, state_color, cumulative_mining)
        
        # Create adaptive legend with icons
        self._draw_adaptive_legend()

    def _draw_adaptive_status_panel(self, state_text, state_color, cumulative_mining) -> None:
        """Draw an adaptive status panel with icons and multi-column layout."""
        try:
            import pygame
            import numpy as np
            from pygame import gfxdraw
        except ImportError:
            return

        # Status data with icons
        status_items = [
            {
                "icon": "energy",
                "value": f"{self.env.agent_energy:.0f}/150",
                "warning": self.env.agent_energy < 30,
                "color": (255, 100, 100) if self.env.agent_energy < 30 else (100, 255, 100)
            },
            {
                "icon": "inventory", 
                "value": f"{self.env.agent_inventory:.0f}/{self.env.max_inventory}",
                "warning": False,
                "color": (255, 255, 100) if self.env.agent_inventory > 0 else (200, 200, 200)
            },
            {
                "icon": "mining",
                "value": f"{cumulative_mining:.1f}",
                "warning": False,
                "color": (100, 255, 100)
            },
            {
                "icon": "collision",
                "value": str(self.env.collision_count),
                "warning": hasattr(self.env, 'last_collision_step') and self.env.last_collision_step == self.env.steps_count,
                "color": (255, 100, 100) if hasattr(self.env, 'last_collision_step') and self.env.last_collision_step == self.env.steps_count else (200, 200, 200)
            },
            {
                "icon": "step",
                "value": f"{self.env.steps_count}/{self.env.max_episode_steps}",
                "warning": False,
                "color": (200, 200, 255)
            },
            {
                "icon": "asteroid",
                "value": f"{np.sum(self.env.asteroid_resources >= 0.1)}/{len(self.env.asteroid_positions)}",
                "warning": False,
                "color": (255, 215, 100)
            }
        ]

        # Calculate adaptive panel size (2 columns, 3 rows)
        cols = 2
        rows = 3
        item_width = 180
        item_height = 40
        panel_width = cols * item_width + 40
        panel_height = rows * item_height + 70

        # Create main status panel
        status_bg = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        status_bg.fill((0, 0, 0, 220))
        pygame.draw.rect(status_bg, (60, 60, 80, 150), (0, 0, panel_width, panel_height), 3)

        # Title with state
        title_font = pygame.font.SysFont("Arial", 16, bold=True)
        title_surface = title_font.render(state_text, True, state_color)
        title_rect = title_surface.get_rect(center=(panel_width // 2, 20))
        status_bg.blit(title_surface, title_rect)

        # Draw status items in grid
        for i, item in enumerate(status_items):
            row = i // cols
            col = i % cols
            x = 15 + col * item_width
            y = 40 + row * item_height

            # Draw icon
            icon_x = x + 5
            icon_y = y + 5
            self._draw_status_icon(status_bg, item["icon"], icon_x, icon_y, item["warning"])

            # Draw value text
            value_font = pygame.font.SysFont("Arial", 16, bold=item["warning"])
            value_surface = value_font.render(item["value"], True, item["color"])
            status_bg.blit(value_surface, (x + 35, y + 10))

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



    def _draw_adaptive_legend(self) -> None:
        """Draw an adaptive legend with icons and optimized layout."""
        try:
            import pygame
            from pygame import gfxdraw
            import math
        except ImportError:
            return

        # Legend items with visual icons
        legend_items = [
            {"icon": "agent", "text": "Mining Agent", "color": (50, 255, 50)},
            {"icon": "mothership", "text": "Mothership", "color": (30, 120, 200)},
            {"icon": "asteroid", "text": "Asteroids", "color": (255, 215, 0)},
            {"icon": "obstacle", "text": "Obstacles", "color": (220, 50, 50)},
            {"icon": "depleted", "text": "Depleted", "color": (100, 100, 100)},
            {"icon": "obs_range", "text": "View Range", "color": (100, 150, 255)},
            {"icon": "mine_range", "text": "Mine Range", "color": (255, 100, 100)},
            {"icon": "safe_zone", "text": "Safe Zone", "color": (30, 120, 255)},
            {"icon": "energy_beam", "text": "Mining", "color": (255, 255, 0)},
            {"icon": "particles", "text": "Delivery", "color": (255, 255, 0)},
            {"icon": "trail", "text": "Agent Trail", "color": (50, 255, 50)},
            {"icon": "dim_area", "text": "Dim Area", "color": (50, 50, 50)}
        ]

        # Calculate adaptive layout (5 columns for larger screen)
        cols = 5
        item_width = 160
        item_height = 32
        panel_width = cols * item_width + 50
        panel_height = len(legend_items) // cols * item_height + 100
        if len(legend_items) % cols != 0:
            panel_height += item_height

        # Create legend background
        legend_bg = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        legend_bg.fill((0, 0, 0, 220))
        pygame.draw.rect(legend_bg, (60, 60, 80, 150), (0, 0, panel_width, panel_height), 3)

        # Title
        title_font = pygame.font.SysFont("Arial", 18, bold=True)
        title_surface = title_font.render("GAME LEGEND", True, (255, 255, 150))
        title_rect = title_surface.get_rect(center=(panel_width // 2, 25))
        legend_bg.blit(title_surface, title_rect)

        # Draw legend items in grid
        for i, item in enumerate(legend_items):
            row = i // cols
            col = i % cols
            x = 15 + col * item_width
            y = 50 + row * item_height

            # Draw icon
            self._draw_legend_icon(legend_bg, item["icon"], x + 8, y + 4, item["color"])

            # Draw text
            text_font = pygame.font.SysFont("Arial", 13)
            text_surface = text_font.render(item["text"], True, (220, 220, 220))
            legend_bg.blit(text_surface, (x + 35, y + 7))

        # Position legend for larger screen
        legend_x = 1440 - panel_width - 20
        legend_y = 1080 - panel_height - 20
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
                alpha = 100 - i * 30
                gfxdraw.aacircle(surface, x+10, y+10, 8 + i * 2, (*color[:3], alpha))

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

    def _draw_floating_timeline(self) -> None:
        """Draw the floating event timeline at the top of the screen."""
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            return

        if not self.env.event_timeline:
            return

        # Timeline positioning for larger screen
        timeline_y = 20
        card_width = 180
        card_height = 40
        card_spacing = 15
        start_x = (1440 - (len(self.env.event_timeline) * (card_width + card_spacing) - card_spacing)) // 2

        for i, event in enumerate(self.env.event_timeline):
            x = start_x + i * (card_width + card_spacing)
            y = timeline_y
            
            # Create micro-card
            alpha = max(50, min(255, event["alpha"]))
            
            # Card background with transparency
            card_surface = pygame.Surface((card_width, card_height), pygame.SRCALPHA)
            
            # Background color based on event type
            if event["type"] == "mining":
                bg_color = (40, 80, 40, alpha)
                border_color = (*event["color"][:3], alpha)
            elif event["type"] == "delivery":
                bg_color = (40, 60, 80, alpha)
                border_color = (*event["color"][:3], alpha)
            elif event["type"] == "collision":
                bg_color = (80, 40, 40, alpha)
                border_color = (*event["color"][:3], alpha)
            elif event["type"] == "combo":
                bg_color = (80, 60, 20, alpha)
                border_color = (*event["color"][:3], alpha)
            else:
                bg_color = (60, 60, 60, alpha)
                border_color = (200, 200, 200, alpha)
            
            # Draw card background
            card_surface.fill(bg_color)
            pygame.draw.rect(card_surface, border_color, (0, 0, card_width, card_height), 2)
            
            # Add icon based on event type
            self._draw_timeline_icon(card_surface, event["type"], 8, 5)
            
            # Add text
            font = pygame.font.SysFont("Arial", 11, bold=True)
            text_surface = font.render(event["text"], True, (*event["color"][:3], alpha))
            
            # Position text to the right of icon
            text_rect = text_surface.get_rect()
            text_x = 28
            text_y = (card_height - text_rect.height) // 2
            card_surface.blit(text_surface, (text_x, text_y))
            
            # Blit card to main window
            self.window.blit(card_surface, (x, y))

    def _draw_timeline_icon(self, surface, event_type, x, y) -> None:
        """Draw small icons for timeline events."""
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            return

        if event_type == "mining":
            # Small pickaxe
            color = (255, 255, 0)
            pygame.draw.line(surface, color, (x+2, y+15), (x+14, y+3), 2)
            pygame.draw.rect(surface, color, (x+11, y+1, 4, 3))
            
        elif event_type == "delivery":
            # Small arrow pointing up
            color = (0, 255, 0)
            points = [(x+8, y+2), (x+12, y+8), (x+10, y+8), (x+10, y+16), (x+6, y+16), (x+6, y+8), (x+4, y+8)]
            pygame.draw.polygon(surface, color, points)
            
        elif event_type == "collision":
            # Warning symbol
            color = (255, 100, 100)
            gfxdraw.filled_circle(surface, x+8, y+10, 7, color)
            font = pygame.font.SysFont("Arial", 10, bold=True)
            text = font.render("!", True, (0, 0, 0))
            surface.blit(text, (x+5, y+4))
            
        elif event_type == "combo":
            # Star/burst symbol
            color = (255, 200, 0)
            center_x, center_y = x+8, y+10
            for i in range(8):
                angle = i * 45 * 3.14159 / 180
                end_x = center_x + 6 * math.cos(angle)
                end_y = center_y + 6 * math.sin(angle)
                pygame.draw.line(surface, color, (center_x, center_y), (end_x, end_y), 2)

    def _draw_combo_display(self) -> None:
        """Draw the combo multiplier display."""
        try:
            import pygame
            from pygame import gfxdraw
            import math
        except ImportError:
            return

        if self.env.combo_state["display_timer"] <= 0 or self.env.combo_state["chain_count"] < 2:
            return

        # Position combo display for larger screen
        combo_x = 720  # Center of larger screen
        combo_y = 140

        # Create pulsing combo badge
        alpha = self.env.combo_state["combo_alpha"]
        combo_text = f"x{self.env.combo_state['chain_count']} COMBO!"
        
        # Large, bold font for combo
        combo_font = pygame.font.SysFont("Arial", 36, bold=True)
        text_surface = combo_font.render(combo_text, True, (255, 200, 0))
        text_rect = text_surface.get_rect()
        
        # Background badge
        badge_width = text_rect.width + 40
        badge_height = text_rect.height + 20
        badge_surface = pygame.Surface((badge_width, badge_height), pygame.SRCALPHA)
        
        # Gradient background effect
        badge_surface.fill((80, 60, 0, alpha))
        pygame.draw.rect(badge_surface, (255, 200, 0, alpha), (0, 0, badge_width, badge_height), 3)
        
        # Add sparkle effects around badge
        for i in range(8):
            angle = i * 45 + self.env.steps_count * 5  # Rotating sparkles
            sparkle_x = combo_x + 50 * math.cos(math.radians(angle))
            sparkle_y = combo_y + 30 * math.sin(math.radians(angle))
            sparkle_color = (255, 255, 100, alpha // 2)
            
            sparkle_surface = pygame.Surface((6, 6), pygame.SRCALPHA)
            gfxdraw.filled_circle(sparkle_surface, 3, 3, 2, sparkle_color)
            self.window.blit(sparkle_surface, (int(sparkle_x) - 3, int(sparkle_y) - 3))
        
        # Position and draw badge
        badge_x = combo_x - badge_width // 2
        badge_y = combo_y - badge_height // 2
        
        # Set alpha for the text
        text_surface.set_alpha(alpha)
        
        # Draw badge background
        self.window.blit(badge_surface, (badge_x, badge_y))
        
        # Draw combo text
        text_x = badge_x + (badge_width - text_rect.width) // 2
        text_y = badge_y + (badge_height - text_rect.height) // 2
        self.window.blit(text_surface, (text_x, text_y))

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
        
        # Create fade overlay for larger screen
        fade_surface = pygame.Surface((1440, 1080), pygame.SRCALPHA)
        fade_surface.fill((0, 0, 0, fade_alpha))
        self.window.blit(fade_surface, (0, 0))

        # Only show text once fade is substantial
        if fade_alpha > 100:
            stats = self.env.game_over_state["final_stats"]
            success = self.env.game_over_state["success"]
            
            # Title for larger screen
            title_font = pygame.font.SysFont("Arial", 64, bold=True)
            title_text = "🎉 MISSION SUCCESS! 🎉" if success else "💥 MISSION FAILED 💥"
            title_color = (0, 255, 0) if success else (255, 100, 100)
            title_surface = title_font.render(title_text, True, title_color)
            title_rect = title_surface.get_rect(center=(720, 220))
            self.window.blit(title_surface, title_rect)

            # Final statistics
            stats_font = pygame.font.SysFont("Arial", 24)
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

            y_offset = 280
            for line in stats_text:
                if line == "":
                    y_offset += 18
                    continue
                    
                color = (255, 255, 100) if "STATISTICS" in line else (255, 255, 255)
                if "restart" in line:
                    color = (200, 200, 255)
                
                text_surface = stats_font.render(line, True, color)
                text_rect = text_surface.get_rect(center=(600, y_offset))
                self.window.blit(text_surface, text_rect)
                y_offset += 35

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
