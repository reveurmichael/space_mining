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
                self.window = pygame.display.set_mode((2560, 1600))  # MASSIVE ultra-wide for ULTIMATE cosmic immersion
                pygame.display.set_caption("🌌 ULTIMATE COSMIC SPACE MINING UNIVERSE - INFINITE GALAXY EXPLORER 🌌")
            else:
                # Off-screen surface for rgb_array mode
                self.window = pygame.Surface((2560, 1600))
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
        def to_screen(pos, scale=18.0):  # Enhanced scale for massive ultra-wide screen
            x, y = pos
            zoom_scale = scale * self.env.zoom_level
            screen_x = int(1280 + (x - 40) * zoom_scale + shake_offset[0])  # Center at 2560/2
            screen_y = int(800 + (y - 40) * zoom_scale + shake_offset[1])  # Center at 1600/2
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
        overlay.fill((0, 0, 0, 80))  # Even less dimming for spectacular cosmic view
        
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
        
        # Draw ULTIMATE spectacular cosmic phenomena
        self._draw_black_holes()  # NEW: Massive gravitational monsters
        self._draw_quasars()  # NEW: Ultra-bright galactic nuclei
        self._draw_cosmic_ribbons()  # NEW: Flowing energy streams
        self._draw_cosmic_storms()
        self._draw_wormholes()
        self._draw_cosmic_auroras()
        self._draw_pulsars()
        self._draw_shooting_stars()
        self._draw_cosmic_lightning()

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
            if x < -size or x > 1920 + size or y < -size or y > 1200 + size:
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
            if x < -size or x > 1920 + size or y < -size or y > 1200 + size:
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
                    
                                         if 0 <= arm_x <= 1920 and 0 <= arm_y <= 1200:
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
            
            if 0 <= x <= 2560 and 0 <= y <= 1600:
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
                
                if 0 <= x <= 2560 and 0 <= y <= 1600:
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

    def _draw_black_holes(self) -> None:
        """Draw spectacular massive black holes with accretion disks and gravitational lensing."""
        try:
            import pygame
            from pygame import gfxdraw
            import math
        except ImportError:
            return

        for black_hole in self.env.black_holes:
            x, y = int(black_hole["x"]), int(black_hole["y"])
            event_horizon = int(black_hole["size"] * self.env.zoom_level)
            accretion_size = int(black_hole["accretion_disk_size"] * self.env.zoom_level)
            
            # Skip if outside screen bounds
            if x < -accretion_size or x > 2560 + accretion_size or y < -accretion_size or y > 1600 + accretion_size:
                continue
            
            # Draw accretion disk (swirling matter)
            rotation = black_hole["rotation"]
            for ring in range(event_horizon + 10, accretion_size, 8):
                ring_alpha = int(150 * black_hole["intensity"] * (1.0 - (ring - event_horizon) / (accretion_size - event_horizon)))
                if ring_alpha > 5:
                    # Create spiral pattern
                    spiral_offset = math.sin(rotation + ring * 0.1) * 20
                    gfxdraw.aacircle(self.window, x + int(spiral_offset), y, ring, (255, 150, 50, ring_alpha))
            
            # Draw gravitational lensing rings
            for i in range(black_hole["gravity_rings"]):
                ring_radius = event_horizon + i * 15
                lensing_alpha = int(80 * black_hole["intensity"] * (1.0 - i / black_hole["gravity_rings"]))
                if lensing_alpha > 3:
                    gfxdraw.aacircle(self.window, x, y, ring_radius, (200, 200, 255, lensing_alpha))
            
            # Draw polar jets
            jet_length = int(black_hole["jet_length"] * self.env.zoom_level)
            jet_angle = black_hole["jet_angle"]
            jet_end_x1 = x + int(math.cos(jet_angle) * jet_length)
            jet_end_y1 = y + int(math.sin(jet_angle) * jet_length)
            jet_end_x2 = x - int(math.cos(jet_angle) * jet_length)
            jet_end_y2 = y - int(math.sin(jet_angle) * jet_length)
            
            # Draw both jets
            for end_x, end_y in [(jet_end_x1, jet_end_y1), (jet_end_x2, jet_end_y2)]:
                pygame.draw.line(self.window, (100, 200, 255), (x, y), (end_x, end_y), 3)
            
            # Draw event horizon (pure black)
            gfxdraw.filled_circle(self.window, x, y, event_horizon, (0, 0, 0))

    def _draw_quasars(self) -> None:
        """Draw ultra-bright quasars with intense beams."""
        try:
            import pygame
            from pygame import gfxdraw
            import math
        except ImportError:
            return

        for quasar in self.env.quasars:
            x, y = int(quasar["x"]), int(quasar["y"])
            core_size = int(quasar["size"] * self.env.zoom_level)
            beam_length = int(quasar["beam_length"] * self.env.zoom_level)
            
            # Skip if outside screen bounds
            if x < -beam_length or x > 2560 + beam_length or y < -beam_length or y > 1600 + beam_length:
                continue
            
            r, g, b = quasar["color"]
            brightness = quasar["brightness"]
            
            # Pulsing effect
            pulse_intensity = math.sin(self.env.cosmic_time * 0.1 + quasar["pulse_offset"]) * 0.3 + 0.7
            
            # Draw brilliant core with multiple glow layers
            for glow_ring in range(8):
                glow_radius = core_size + glow_ring * 8
                glow_alpha = int(255 * brightness * pulse_intensity * (1.0 - glow_ring / 8))
                if glow_alpha > 10:
                    gfxdraw.aacircle(self.window, x, y, glow_radius, (r, g, b, glow_alpha))
            
            # Draw super-bright core
            gfxdraw.filled_circle(self.window, x, y, core_size, (r, g, b))
            
            # Draw directional beam
            beam_angle = quasar["beam_angle"]
            beam_width = int(quasar["beam_width"] * self.env.zoom_level)
            beam_end_x = x + int(math.cos(beam_angle) * beam_length)
            beam_end_y = y + int(math.sin(beam_angle) * beam_length)
            
            # Draw beam as a series of overlapping circles for smooth gradient
            for i in range(0, beam_length, 20):
                beam_x = x + int(math.cos(beam_angle) * i)
                beam_y = y + int(math.sin(beam_angle) * i)
                beam_alpha = int(120 * brightness * pulse_intensity * (1.0 - i / beam_length))
                if beam_alpha > 5:
                    beam_radius = max(2, int(beam_width * (1.0 - i / beam_length)))
                    gfxdraw.filled_circle(self.window, beam_x, beam_y, beam_radius, (r, g, b, beam_alpha))

    def _draw_cosmic_ribbons(self) -> None:
        """Draw flowing cosmic energy ribbons."""
        try:
            import pygame
            from pygame import gfxdraw
            import math
        except ImportError:
            return

        for ribbon in self.env.cosmic_ribbons:
            if len(ribbon["points"]) < 2:
                continue
                
            width = int(ribbon["width"] * self.env.zoom_level)
            r, g, b, base_alpha = ribbon["color"]
            
            # Draw ribbon as connected segments
            for i in range(len(ribbon["points"]) - 1):
                p1 = ribbon["points"][i]
                p2 = ribbon["points"][i + 1]
                
                x1, y1 = int(p1[0]), int(p1[1])
                x2, y2 = int(p2[0]), int(p2[1])
                
                # Skip if outside screen bounds
                if (x1 < -width and x2 < -width) or (x1 > 2560 + width and x2 > 2560 + width):
                    continue
                if (y1 < -width and y2 < -width) or (y1 > 1600 + width and y2 > 1600 + width):
                    continue
                
                # Add flowing wave effect
                wave_offset = math.sin(self.env.cosmic_time * ribbon["wave_frequency"] + i * 0.5) * ribbon["wave_amplitude"]
                
                # Calculate perpendicular offset for wave effect
                dx = x2 - x1
                dy = y2 - y1
                length = math.sqrt(dx*dx + dy*dy)
                if length > 0:
                    perp_x = -dy / length * wave_offset
                    perp_y = dx / length * wave_offset
                    
                    wave_x1 = x1 + int(perp_x)
                    wave_y1 = y1 + int(perp_y)
                    wave_x2 = x2 + int(perp_x)
                    wave_y2 = y2 + int(perp_y)
                    
                    # Draw ribbon segment with multiple layers for glow effect
                    for layer in range(3):
                        layer_width = width - layer * 3
                        layer_alpha = int(base_alpha * (1.0 - layer * 0.3))
                        if layer_width > 0 and layer_alpha > 5:
                            pygame.draw.line(self.window, (r, g, b, layer_alpha), 
                                           (wave_x1, wave_y1), (wave_x2, wave_y2), layer_width)

    def _draw_cosmic_auroras(self) -> None:
        """Draw ethereal cosmic auroras."""
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            return

        for aurora in self.env.cosmic_auroras:
            x, y = int(aurora["x"]), int(aurora["y"])
            width = int(aurora["width"] * self.env.zoom_level)
            height = int(aurora["height"] * self.env.zoom_level)
            
            # Skip if off screen
            if x < -width or x > 1920 + width or y < -height or y > 1200 + height:
                continue
            
            r, g, b, base_alpha = aurora["color"]
            intensity = aurora["intensity"]
            wave_offset = aurora["wave_offset"]
            
            # Create wavy aurora effect
            aurora_surface = pygame.Surface((width, height), pygame.SRCALPHA)
            
            # Draw vertical aurora strips with wave motion
            for strip_x in range(0, width, 8):
                wave_y = int(math.sin(wave_offset + strip_x * 0.05) * 30)
                strip_height = height + wave_y
                
                # Gradient from top to bottom
                for strip_y in range(max(0, -wave_y), min(height, strip_height)):
                    if 0 <= strip_y < height:
                        # Fade towards edges
                        edge_fade = 1.0 - abs(strip_x - width//2) / (width//2)
                        edge_fade *= 1.0 - abs(strip_y - height//2) / (height//2)
                        
                        alpha = int(base_alpha * intensity * edge_fade)
                        if alpha > 3:
                            color = (r, g, b, alpha)
                            for dx in range(8):
                                if strip_x + dx < width:
                                    aurora_surface.set_at((strip_x + dx, strip_y), color)
            
            # Blit aurora to main surface
            self.window.blit(aurora_surface, (x, y))

    def _draw_pulsars(self) -> None:
        """Draw spectacular pulsing neutron stars."""
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            return

        for pulsar in self.env.pulsars:
            x, y = int(pulsar["x"]), int(pulsar["y"])
            
            if 0 <= x <= 2560 and 0 <= y <= 1600:
                # Pulse calculation
                pulse_phase = math.sin(self.env.cosmic_time / pulsar["pulse_period"] + pulsar["pulse_offset"])
                pulse_intensity = (pulse_phase + 1) / 2  # 0 to 1
                
                brightness = int(pulsar["brightness"] * pulse_intensity)
                r, g, b = pulsar["color"]
                
                # Draw pulsing core
                core_size = max(2, int(6 * pulse_intensity * self.env.zoom_level))
                core_color = (min(255, int(r * pulse_intensity)), 
                             min(255, int(g * pulse_intensity)), 
                             min(255, int(b * pulse_intensity)))
                
                gfxdraw.filled_circle(self.window, x, y, core_size, core_color)
                
                # Draw rotating beam when pulsing
                if pulse_intensity > 0.7:  # Only when bright
                    beam_length = int(pulsar["beam_length"] * self.env.zoom_level)
                    beam_angle = pulsar["beam_angle"]
                    
                    # Draw beam as series of fading dots
                    for r in range(10, beam_length, 15):
                        beam_x = x + int(r * math.cos(beam_angle))
                        beam_y = y + int(r * math.sin(beam_angle))
                        
                        if 0 <= beam_x <= 2560 and 0 <= beam_y <= 1600:
                            beam_fade = 1.0 - (r / beam_length)
                            beam_alpha = int(brightness * beam_fade * 0.8)
                            if beam_alpha > 10:
                                beam_color = (beam_alpha, beam_alpha, beam_alpha + 20)
                                beam_size = max(1, int(3 * beam_fade))
                                if beam_size > 1:
                                    gfxdraw.filled_circle(self.window, beam_x, beam_y, beam_size, beam_color)
                                else:
                                    self.window.set_at((beam_x, beam_y), beam_color)

    def _draw_shooting_stars(self) -> None:
        """Draw spectacular shooting stars with trails."""
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            return

        for star in self.env.shooting_stars:
            x, y = int(star["x"]), int(star["y"])
            
            if 0 <= x <= 2560 and 0 <= y <= 1600:
                r, g, b = star["color"]
                brightness = star["brightness"]
                tail_length = star["tail_length"]
                
                # Calculate direction for tail
                dx = star["target_x"] - star["x"]
                dy = star["target_y"] - star["y"]
                if dx != 0 or dy != 0:
                    distance = math.sqrt(dx*dx + dy*dy)
                    tail_dx = -(dx / distance) * 15  # Opposite direction
                    tail_dy = -(dy / distance) * 15
                    
                    # Draw fading tail
                    for i in range(tail_length):
                        tail_x = int(x + tail_dx * i / 2)
                        tail_y = int(y + tail_dy * i / 2)
                        
                        if 0 <= tail_x <= 2560 and 0 <= tail_y <= 1600:
                            fade = 1.0 - (i / tail_length)
                            tail_brightness = int(brightness * fade)
                            if tail_brightness > 5:
                                tail_color = (
                                    min(255, int(r * fade)),
                                    min(255, int(g * fade)),
                                    min(255, int(b * fade))
                                )
                                
                                tail_size = max(1, int(3 * fade))
                                if tail_size > 1:
                                    gfxdraw.filled_circle(self.window, tail_x, tail_y, tail_size, tail_color)
                                else:
                                    self.window.set_at((tail_x, tail_y), tail_color)
                
                # Draw bright star head
                star_color = (r, g, b)
                star_size = max(3, int(5 * self.env.zoom_level))
                gfxdraw.filled_circle(self.window, x, y, star_size, star_color)
                
                # Add bright glow
                glow_color = (r//2, g//2, b//2)
                gfxdraw.aacircle(self.window, x, y, star_size + 2, glow_color)

    def _draw_cosmic_storms(self) -> None:
        """Draw spectacular rotating cosmic storms with lightning."""
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            return

        for storm in self.env.cosmic_storms:
            x, y = int(storm["x"]), int(storm["y"])
            size = int(storm["size"] * self.env.zoom_level)
            
            if x < -size or x > 1920 + size or y < -size or y > 1200 + size:
                continue
            
            r, g, b, base_alpha = storm["color"]
            intensity = storm["intensity"]
            rotation = storm["rotation"]
            
            # Create swirling storm effect
            storm_surface = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            center = size
            
            # Draw storm layers with rotation
            for layer in range(8):
                layer_radius = int(size * (1.0 - layer * 0.12))
                if layer_radius > 5:
                    alpha = int(base_alpha * intensity * (1.0 - layer * 0.1))
                    
                    # Spiral arms of the storm
                    for arm in range(4):
                        arm_angle = rotation + (arm * math.pi / 2)
                        
                        for r in range(10, layer_radius, 8):
                            spiral_angle = arm_angle + r * 0.1
                            
                            storm_x = center + int(r * math.cos(spiral_angle))
                            storm_y = center + int(r * math.sin(spiral_angle))
                            
                            if 0 <= storm_x < size * 2 and 0 <= storm_y < size * 2:
                                fade = 1.0 - (r / layer_radius)
                                storm_alpha = int(alpha * fade)
                                
                                if storm_alpha > 5:
                                    storm_color = (
                                        min(255, int(r * intensity)),
                                        min(255, int(g * intensity)),
                                        min(255, int(b * intensity)),
                                        storm_alpha
                                    )
                                    
                                    # Draw storm particles
                                    particle_size = max(1, int(3 * fade))
                                    if particle_size > 1:
                                        gfxdraw.filled_circle(storm_surface, storm_x, storm_y, particle_size, storm_color)
                                    else:
                                        storm_surface.set_at((storm_x, storm_y), storm_color)
            
            # Blit storm to main surface
            storm_rect = storm_surface.get_rect(center=(x, y))
            self.window.blit(storm_surface, storm_rect)

    def _draw_wormholes(self) -> None:
        """Draw mystical dimensional wormholes."""
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            return

        for wormhole in self.env.wormholes:
            x, y = int(wormhole["x"]), int(wormhole["y"])
            size = int(wormhole["size"] * self.env.zoom_level)
            
            if x < -size or x > 1920 + size or y < -size or y > 1200 + size:
                continue
            
            rotation = wormhole["rotation"]
            rings = wormhole["distortion_rings"]
            
            # Pulsing effect
            pulse = math.sin(self.env.cosmic_time * 2 + wormhole["pulse_offset"]) * 0.3 + 0.7
            
            # Draw concentric distortion rings
            for ring in range(rings):
                ring_radius = int(size * (1.0 - ring * 0.12) * pulse)
                if ring_radius > 2:
                    # Alternating colors for mystical effect
                    if ring % 2 == 0:
                        ring_color = (100, 0, 200, 150 - ring * 15)  # Purple
                    else:
                        ring_color = (0, 150, 255, 120 - ring * 12)  # Blue
                    
                    # Create ring with rotation distortion
                    ring_surface = pygame.Surface((ring_radius * 2 + 10, ring_radius * 2 + 10), pygame.SRCALPHA)
                    ring_center = ring_radius + 5
                    
                    # Draw distorted ring
                    for angle_deg in range(0, 360, 6):
                        angle_rad = math.radians(angle_deg + rotation * 180 / math.pi)
                        
                        # Distortion effect
                        distort = math.sin(angle_rad * 4 + self.env.cosmic_time) * 0.2 + 1.0
                        actual_radius = int(ring_radius * distort)
                        
                        ring_x = ring_center + int(actual_radius * math.cos(angle_rad))
                        ring_y = ring_center + int(actual_radius * math.sin(angle_rad))
                        
                        if 0 <= ring_x < ring_radius * 2 + 10 and 0 <= ring_y < ring_radius * 2 + 10:
                            gfxdraw.filled_circle(ring_surface, ring_x, ring_y, 2, ring_color)
                    
                    # Blit ring to main surface
                    ring_rect = ring_surface.get_rect(center=(x, y))
                    self.window.blit(ring_surface, ring_rect)
            
            # Draw bright center
            center_size = max(3, int(8 * pulse * self.env.zoom_level))
            center_color = (255, 255, 255, 200)
            gfxdraw.filled_circle(self.window, x, y, center_size, center_color)

    def _draw_cosmic_lightning(self) -> None:
        """Draw spectacular branching cosmic lightning."""
        try:
            import pygame
        except ImportError:
            return

        for lightning in self.env.cosmic_lightning:
            if lightning["intensity"] < 0.1:
                continue
            
            r, g, b = lightning["color"]
            intensity = lightning["intensity"]
            thickness = max(1, int(lightning["thickness"] * intensity))
            
            # Main lightning bolt
            start_pos = (int(lightning["start_x"]), int(lightning["start_y"]))
            end_pos = (int(lightning["end_x"]), int(lightning["end_y"]))
            
            if (0 <= start_pos[0] <= 1920 and 0 <= start_pos[1] <= 1200 and
                0 <= end_pos[0] <= 1920 and 0 <= end_pos[1] <= 1200):
                
                lightning_color = (
                    min(255, int(r * intensity)),
                    min(255, int(g * intensity)),
                    min(255, int(b * intensity))
                )
                
                # Draw main bolt with multiple lines for thickness
                for i in range(thickness):
                    offset_x = np.random.randint(-1, 2)
                    offset_y = np.random.randint(-1, 2)
                    
                    adjusted_start = (start_pos[0] + offset_x, start_pos[1] + offset_y)
                    adjusted_end = (end_pos[0] + offset_x, end_pos[1] + offset_y)
                    
                    pygame.draw.line(self.window, lightning_color, adjusted_start, adjusted_end, 1)
                
                # Draw branches
                for branch in lightning["branches"]:
                    branch_start = (int(branch["start_x"]), int(branch["start_y"]))
                    branch_end = (int(branch["end_x"]), int(branch["end_y"]))
                    
                    if (0 <= branch_start[0] <= 1920 and 0 <= branch_start[1] <= 1200 and
                        0 <= branch_end[0] <= 1920 and 0 <= branch_end[1] <= 1200):
                        
                        branch_color = (
                            min(255, int(r * intensity * 0.7)),
                            min(255, int(g * intensity * 0.7)),
                            min(255, int(b * intensity * 0.7))
                        )
                        
                        pygame.draw.line(self.window, branch_color, branch_start, branch_end, max(1, thickness // 2))

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

        # Streamlined essential status only
        status_items = [
            {
                "icon": "energy",
                "value": f"Energy {self.env.agent_energy:.0f}",
                "warning": self.env.agent_energy < 30,
                "color": (255, 100, 100) if self.env.agent_energy < 30 else (100, 255, 100)
            },
            {
                "icon": "inventory", 
                "value": f"Cargo {self.env.agent_inventory:.1f}",
                "warning": False,
                "color": (255, 255, 100) if self.env.agent_inventory > 0 else (200, 200, 200)
            },
            {
                "icon": "mining",
                "value": f"Mined {cumulative_mining:.1f}",
                "warning": False,
                "color": (100, 255, 100)
            },
            {
                "icon": "asteroid",
                "value": f"Asteroids {np.sum(self.env.asteroid_resources >= 0.1)}",
                "warning": np.sum(self.env.asteroid_resources >= 0.1) <= 2,
                "color": (255, 215, 100)
            }
        ]

        # Calculate streamlined panel size (2 columns, 2 rows - removed redundant info)
        cols = 2
        rows = 2
        item_width = 200
        item_height = 45
        panel_width = cols * item_width + 50
        panel_height = rows * item_height + 80

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

        # Calculate adaptive layout (6 columns for ultra-wide screen)
        cols = 6
        item_width = 180
        item_height = 36
        panel_width = cols * item_width + 60
        panel_height = len(legend_items) // cols * item_height + 120
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

        # Position legend for ultra-wide screen
        legend_x = 1920 - panel_width - 25
        legend_y = 1200 - panel_height - 25
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

        # Timeline positioning for ultra-wide screen
        timeline_y = 25
        card_width = 200
        card_height = 45
        card_spacing = 18
        start_x = (1920 - (len(self.env.event_timeline) * (card_width + card_spacing) - card_spacing)) // 2

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

        # Position combo display for ultra-wide screen
        combo_x = 960  # Center of ultra-wide screen
        combo_y = 160

        # Create pulsing combo badge
        alpha = self.env.combo_state["combo_alpha"]
        combo_text = f"x{self.env.combo_state['chain_count']} COMBO!"
        
        # Large, bold font for combo
        combo_font = pygame.font.SysFont("Arial", 40, bold=True)
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
