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
                self.window = pygame.display.set_mode((800, 800))
                pygame.display.set_caption("Space Mining Environment")
            else:
                # Off-screen surface for rgb_array mode
                self.window = pygame.Surface((800, 800))
            if self.clock is None:
                self.clock = pygame.time.Clock()
            if self.font is None:
                pygame.font.init()
                self.font = pygame.font.SysFont("Arial", 16)

        # Screen shake offset
        shake_offset = [0, 0]
        if self.env.screen_shake_timer > 0:
            shake_intensity = min(5, int(self.env.screen_shake_timer * 25))
            shake_offset[0] = random.randint(-shake_intensity, shake_intensity)
            shake_offset[1] = random.randint(-shake_intensity, shake_intensity)

        # Deep space background
        self.window.fill((5, 5, 15))

        # Draw dynamic starfield background
        self._draw_starfield()

        # Helper function to convert 2D coordinates to screen coordinates
        def to_screen(pos, scale=8.0):
            x, y = pos
            screen_x = int(400 + (x - 40) * scale + shake_offset[0])
            screen_y = int(400 + (y - 40) * scale + shake_offset[1])
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
        safe_zone_radius = 12 * 8  # Safe zone is 12 units in game coordinates
        
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
            self.window, mothership_pos_2d[0], mothership_pos_2d[1], 16, (30, 120, 200)
        )
        # Enhanced mothership glow effect
        for i in range(6):
            alpha = max(20, 150 - i * 25)
            gfxdraw.aacircle(
                self.window, 
                mothership_pos_2d[0], 
                mothership_pos_2d[1], 
                16 + i * 3, 
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
            size = int(8 + resource_ratio * 8)
            
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
            12,
            agent_color,
        )

        # Draw white outline for visibility
        gfxdraw.aacircle(
            self.window,
            agent_pos_2d[0],
            agent_pos_2d[1],
            12,
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
        energy_width = int(60 * energy_ratio)
        bar_x = agent_pos_2d[0] - 30
        bar_y = agent_pos_2d[1] - 28
        
        # Background with border
        pygame.draw.rect(self.window, (20, 20, 20), (bar_x - 2, bar_y - 2, 64, 10))
        pygame.draw.rect(self.window, (60, 60, 60), (bar_x, bar_y, 60, 6))
        
        # Energy bar with enhanced colors
        if energy_ratio > 0.6:
            energy_color = (0, 255, 0)
        elif energy_ratio > 0.3:
            energy_color = (255, 255, 0)
        else:
            energy_color = (255, 50, 50)
        pygame.draw.rect(self.window, energy_color, (bar_x, bar_y, energy_width, 6))

        # Draw observation and mining ranges
        obs_radius_px = int(self.env.observation_radius * 8.0)
        mining_radius_px = int(self.env.mining_range * 8.0)
        
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
        overlay = pygame.Surface((800, 800), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        
        # Create a "visible" hole for observation range
        pygame.draw.circle(overlay, (0, 0, 0, 0), agent_pos_2d, obs_radius_px)
        
        # Apply the dimming overlay
        self.window.blit(overlay, (0, 0))

        # Draw collision flash overlay (after dimming so it's always visible)
        if self.env.collision_flash_timer > 0:
            flash_alpha = int(200 * (self.env.collision_flash_timer / 0.3))
            flash_surface = pygame.Surface((800, 800), pygame.SRCALPHA)
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
        """Draw the parallax starfield background."""
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            return

        for layer in self.env.starfield_layers:
            for star in layer:
                x, y = int(star["x"]), int(star["y"])
                if 0 <= x <= 800 and 0 <= y <= 800:
                    size = star["size"]
                    brightness = star["brightness"]
                    color = (brightness, brightness, brightness)
                    
                    if size == 1:
                        # Single pixel star
                        self.window.set_at((x, y), color)
                    else:
                        # Larger stars with glow
                        gfxdraw.filled_circle(self.window, x, y, size, color)
                        if size > 2:
                            # Add subtle glow for close stars
                            glow_color = (brightness // 2, brightness // 2, brightness // 2)
                            gfxdraw.aacircle(self.window, x, y, size + 1, glow_color)

    def _draw_game_ui(self, agent_pos_2d) -> None:
        """Draw the main game UI elements."""
        try:
            import pygame
            import numpy as np
        except ImportError:
            return

        # Enhanced status panel with better layout
        cumulative_mining = getattr(self.env, "cumulative_mining_amount", 0.0)
        
        # Agent state indicator
        state_indicator = "🔍 EXPLORING"
        if hasattr(self.env, "mining_asteroid_id"):
            state_indicator = f"🔨 MINING Asteroid {self.env.mining_asteroid_id}"
        elif self.env.agent_inventory > 0:
            state_indicator = f"📦 CARRYING {self.env.agent_inventory:.1f} resources"
        
        status_text = [
            f"Energy: {self.env.agent_energy:.0f}/150 {'⚠️' if self.env.agent_energy < 30 else '⚡'}",
            f"Inventory: {self.env.agent_inventory:.0f}/{self.env.max_inventory}",
            f"Total Mined: {cumulative_mining:.1f}",
            f"Collisions: {self.env.collision_count} {'💥' if hasattr(self.env, 'last_collision_step') and self.env.last_collision_step == self.env.steps_count else ''}",
            f"Step: {self.env.steps_count}/{self.env.max_episode_steps}",
            f"Asteroids: {np.sum(self.env.asteroid_resources >= 0.1)}/{len(self.env.asteroid_positions)}",
            "",
            state_indicator
        ]

        # Add recent events
        if (hasattr(self.env, "last_mining_info") and self.env.last_mining_info.get("step", 0) == self.env.steps_count):
            extracted = self.env.last_mining_info['extracted']
            status_text.append(f"⛏️ Mined {extracted:.1f} resources!")
            if self.env.last_mining_info.get("asteroid_depleted", False):
                status_text.append("💀 Asteroid depleted!")

        if (hasattr(self.env, "last_delivery_info") and self.env.last_delivery_info.get("step", 0) == self.env.steps_count):
            delivered = self.env.last_delivery_info['delivered']
            status_text.append(f"🚀 Delivered {delivered:.1f} resources!")
            status_text.append("⚡ Fully recharged!")

        if hasattr(self.env, "tried_depleted_asteroid") and self.env.tried_depleted_asteroid:
            status_text.append("⚠️ Tried mining depleted asteroid!")

        # Enhanced status panel with better background
        panel_height = len(status_text) * 24 + 20
        status_bg = pygame.Surface((350, panel_height), pygame.SRCALPHA)
        status_bg.fill((0, 0, 0, 200))
        pygame.draw.rect(status_bg, (50, 50, 50, 100), (0, 0, 350, panel_height), 2)
        self.window.blit(status_bg, (10, 10))
        
        for i, text in enumerate(status_text):
            color = (255, 255, 255) if text != "" else (200, 200, 200)
            if "⚠️" in text or "💥" in text:
                color = (255, 200, 200)
            elif "✅" in text or "🚀" in text or "⚡" in text:
                color = (200, 255, 200)
            
            rendered_text = self.font.render(text, True, color)
            self.window.blit(rendered_text, (20, 20 + i * 24))

        # Enhanced legend with horizontal layout optimization
        legend_text = [
            "🎮 GAME LEGEND",
            "",
            "🤖 Green Circle = Mining Agent (always green)",
            "🏭 Blue Circle = Mothership (recharge & delivery)",
            "🌕 Yellow Circles = Asteroids (size = resources)",
            "💀 Gray Circle + X = Depleted Asteroids",
            "☄️ Red Pulsing = Moving Obstacles (danger!)",
            "",
            "🔵 Blue Ring = Observation Range",
            "🔴 Red Ring = Mining Range", 
            "⚡ Yellow Beam = Active Mining Energy",
            "✨ Yellow Dots = Resource Delivery",
            "🌑 Dim Area = Outside Observation Range",
            "💫 Blue Aura = Mothership Safe Zone"
        ]
        
        legend_height = len(legend_text) * 22 + 20
        legend_bg = pygame.Surface((420, legend_height), pygame.SRCALPHA)
        legend_bg.fill((0, 0, 0, 200))
        pygame.draw.rect(legend_bg, (50, 50, 50, 100), (0, 0, 420, legend_height), 2)
        self.window.blit(legend_bg, (370, 580 - legend_height))
        
        for i, text in enumerate(legend_text):
            if text == "":
                continue
            color = (255, 255, 100) if i == 0 else (220, 220, 220)
            font_to_use = pygame.font.SysFont("Arial", 14, bold=(i == 0))
            rendered_text = font_to_use.render(text, True, color)
            self.window.blit(rendered_text, (380, 590 - legend_height + i * 22))

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
        
        # Create fade overlay
        fade_surface = pygame.Surface((800, 800), pygame.SRCALPHA)
        fade_surface.fill((0, 0, 0, fade_alpha))
        self.window.blit(fade_surface, (0, 0))

        # Only show text once fade is substantial
        if fade_alpha > 100:
            stats = self.env.game_over_state["final_stats"]
            success = self.env.game_over_state["success"]
            
            # Title
            title_font = pygame.font.SysFont("Arial", 48, bold=True)
            title_text = "🎉 MISSION SUCCESS! 🎉" if success else "💥 MISSION FAILED 💥"
            title_color = (0, 255, 0) if success else (255, 100, 100)
            title_surface = title_font.render(title_text, True, title_color)
            title_rect = title_surface.get_rect(center=(400, 150))
            self.window.blit(title_surface, title_rect)

            # Final statistics
            stats_font = pygame.font.SysFont("Arial", 20)
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

            y_offset = 220
            for line in stats_text:
                if line == "":
                    y_offset += 15
                    continue
                    
                color = (255, 255, 100) if "STATISTICS" in line else (255, 255, 255)
                if "restart" in line:
                    color = (200, 200, 255)
                
                text_surface = stats_font.render(line, True, color)
                text_rect = text_surface.get_rect(center=(400, y_offset))
                self.window.blit(text_surface, text_rect)
                y_offset += 30

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
