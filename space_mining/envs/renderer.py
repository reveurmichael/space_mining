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

        self.window.fill((0, 0, 0))  # Space background (black)

        # Helper function to convert 2D coordinates to screen coordinates
        def to_screen(pos, scale=8.0):
            x, y = pos
            # Center the 80x80 grid in the 800x800 screen
            # Map 0-80 to 0-800, centered at 400
            screen_x = int(400 + (x - 40) * scale + shake_offset[0])
            screen_y = int(400 + (y - 40) * scale + shake_offset[1])
            return screen_x, screen_y

        # Draw agent trail first (behind everything)
        for trail_point in self.env.agent_trail:
            trail_pos_2d = to_screen(trail_point["pos"])
            alpha = max(0, min(255, trail_point["alpha"]))
            if alpha > 0:
                # Create a surface for the trail point with alpha
                trail_surface = pygame.Surface((8, 8), pygame.SRCALPHA)
                trail_color = (50, 200, 50, alpha)
                gfxdraw.filled_circle(trail_surface, 4, 4, 3, trail_color)
                self.window.blit(trail_surface, (trail_pos_2d[0] - 4, trail_pos_2d[1] - 4))

        # Draw mothership
        mothership_pos_2d = to_screen(self.env.mothership_pos)
        gfxdraw.filled_circle(
            self.window, mothership_pos_2d[0], mothership_pos_2d[1], 15, (50, 150, 200)
        )
        # Add mothership glow effect
        for i in range(3):
            gfxdraw.aacircle(
                self.window, 
                mothership_pos_2d[0], 
                mothership_pos_2d[1], 
                15 + i * 2, 
                (50, 150, 200, 100 - i * 30)
            )

        # Draw asteroids with health bars
        for i, pos in enumerate(self.env.asteroid_positions):
            if self.env.asteroid_resources[i] < 0.1:  # Depletion threshold
                # Draw depleted asteroid as gray cross
                asteroid_pos_2d = to_screen(pos)
                gfxdraw.filled_circle(
                    self.window,
                    asteroid_pos_2d[0],
                    asteroid_pos_2d[1],
                    8,
                    (100, 100, 100),  # Gray for depleted
                )
                # Draw X mark for depleted asteroid
                pygame.draw.line(
                    self.window,
                    (150, 150, 150),
                    (asteroid_pos_2d[0] - 5, asteroid_pos_2d[1] - 5),
                    (asteroid_pos_2d[0] + 5, asteroid_pos_2d[1] + 5),
                    2,
                )
                pygame.draw.line(
                    self.window,
                    (150, 150, 150),
                    (asteroid_pos_2d[0] + 5, asteroid_pos_2d[1] - 5),
                    (asteroid_pos_2d[0] - 5, asteroid_pos_2d[1] + 5),
                    2,
                )
                continue

            asteroid_pos_2d = to_screen(pos)
            size = int(6 + self.env.asteroid_resources[i] / 8)
            color_intensity = min(255, int(120 + self.env.asteroid_resources[i] * 3))
            gfxdraw.filled_circle(
                self.window,
                asteroid_pos_2d[0],
                asteroid_pos_2d[1],
                size,
                (color_intensity, color_intensity // 2, 20),
            )

            # Draw health bar for asteroid
            health_ratio = self.env.asteroid_resources[i] / 40.0
            health_width = int(24 * health_ratio)
            health_x = asteroid_pos_2d[0] - 12
            health_y = asteroid_pos_2d[1] - size - 10

            # Background (dark gray)
            pygame.draw.rect(self.window, (60, 60, 60), (health_x, health_y, 24, 5))
            # Health bar (green to red based on remaining resources)
            if health_ratio > 0.6:
                health_color = (0, 255, 0)
            elif health_ratio > 0.3:
                health_color = (255, 255, 0)
            else:
                health_color = (255, 0, 0)
            pygame.draw.rect(
                self.window, health_color, (health_x, health_y, health_width, 5)
            )

        # Draw obstacles with pulsing effect
        for pos in self.env.obstacle_positions:
            obstacle_pos_2d = to_screen(pos)
            pulse = int(7 + 2 * math.sin(self.env.steps_count * 0.1))
            gfxdraw.filled_circle(
                self.window, obstacle_pos_2d[0], obstacle_pos_2d[1], pulse, (200, 50, 50)
            )
            # Add danger glow
            gfxdraw.aacircle(
                self.window, obstacle_pos_2d[0], obstacle_pos_2d[1], pulse + 3, (255, 100, 100)
            )

        # Draw agent
        agent_pos_2d = to_screen(self.env.agent_position)

        # Draw agent body - change color based on state
        agent_color = (50, 200, 50)  # Default green
        if hasattr(self.env, "mining_asteroid_id"):
            agent_color = (255, 165, 0)  # Orange when mining
        elif self.env.agent_inventory > 0:
            agent_color = (255, 255, 0)  # Yellow when carrying resources

        gfxdraw.filled_circle(
            self.window,
            agent_pos_2d[0],
            agent_pos_2d[1],
            12,
            agent_color,
        )

        # Draw white outline
        gfxdraw.aacircle(
            self.window,
            agent_pos_2d[0],
            agent_pos_2d[1],
            12,
            (255, 255, 255),
        )

        # Draw animated mining beam
        if hasattr(self.env, "mining_asteroid_id"):
            asteroid_pos_2d = to_screen(
                self.env.asteroid_positions[self.env.mining_asteroid_id]
            )
            
            # Draw animated dashed mining beam
            beam_length = math.sqrt((asteroid_pos_2d[0] - agent_pos_2d[0])**2 + (asteroid_pos_2d[1] - agent_pos_2d[1])**2)
            num_segments = int(beam_length / 8)
            
            for i in range(num_segments):
                t1 = (i + self.env.mining_beam_offset % 1) / num_segments
                t2 = (i + 0.5 + self.env.mining_beam_offset % 1) / num_segments
                
                if t1 <= 1.0 and t2 <= 1.0:
                    x1 = int(agent_pos_2d[0] + t1 * (asteroid_pos_2d[0] - agent_pos_2d[0]))
                    y1 = int(agent_pos_2d[1] + t1 * (asteroid_pos_2d[1] - agent_pos_2d[1]))
                    x2 = int(agent_pos_2d[0] + t2 * (asteroid_pos_2d[0] - agent_pos_2d[0]))
                    y2 = int(agent_pos_2d[1] + t2 * (asteroid_pos_2d[1] - agent_pos_2d[1]))
                    
                    # Alternate colors for energy effect
                    color = (255, 255, 0) if i % 2 == 0 else (255, 200, 0)
                    pygame.draw.line(self.window, color, (x1, y1), (x2, y2), 4)

        # Draw inventory indicator
        if self.env.agent_inventory > 0:
            gfxdraw.filled_circle(
                self.window,
                agent_pos_2d[0],
                agent_pos_2d[1],
                int(6 + self.env.agent_inventory / 8),
                (200, 200, 0),
            )

        # Draw energy bar with better styling
        energy_ratio = self.env.agent_energy / 150.0
        energy_width = int(50 * energy_ratio)
        bar_x = agent_pos_2d[0] - 25
        bar_y = agent_pos_2d[1] - 25
        
        # Background
        pygame.draw.rect(self.window, (40, 40, 40), (bar_x, bar_y, 50, 6))
        # Energy bar with color based on level
        if energy_ratio > 0.6:
            energy_color = (0, 255, 0)
        elif energy_ratio > 0.3:
            energy_color = (255, 255, 0)
        else:
            energy_color = (255, 0, 0)
        pygame.draw.rect(self.window, energy_color, (bar_x, bar_y, energy_width, 6))

        # Draw observation and mining ranges with better styling
        gfxdraw.aacircle(
            self.window,
            agent_pos_2d[0],
            agent_pos_2d[1],
            int(self.env.observation_radius * 8.0),
            (100, 100, 255),
        )
        gfxdraw.aacircle(
            self.window,
            agent_pos_2d[0],
            agent_pos_2d[1],
            int(self.env.mining_range * 8.0),
            (255, 100, 100),
        )

        # Draw delivery particles
        for particle in self.env.delivery_particles:
            progress = particle["progress"]
            # Interpolate position
            current_pos = (
                particle["start_pos"] + progress * (particle["target_pos"] - particle["start_pos"])
            )
            particle_pos_2d = to_screen(current_pos)
            
            # Fade out as particles get closer to target
            alpha = int(255 * (1 - progress * 0.7))
            if alpha > 0:
                # Create glowing particle effect
                particle_surface = pygame.Surface((12, 12), pygame.SRCALPHA)
                glow_color = (255, 255, 0, alpha)
                gfxdraw.filled_circle(particle_surface, 6, 6, 4, glow_color)
                gfxdraw.filled_circle(particle_surface, 6, 6, 2, (255, 255, 255, alpha))
                self.window.blit(particle_surface, (particle_pos_2d[0] - 6, particle_pos_2d[1] - 6))

        # Draw score popups
        for popup in self.env.score_popups:
            popup_pos_2d = to_screen(popup["pos"])
            alpha = max(0, min(255, popup["alpha"]))
            if alpha > 0:
                # Create text surface with alpha
                text_surface = self.font.render(popup["text"], True, popup["color"])
                text_surface.set_alpha(alpha)
                self.window.blit(text_surface, (popup_pos_2d[0] - 20, popup_pos_2d[1] - 10))

        # Draw collision flash overlay
        if self.env.collision_flash_timer > 0:
            flash_alpha = int(180 * (self.env.collision_flash_timer / 0.3))
            flash_surface = pygame.Surface((800, 800), pygame.SRCALPHA)
            flash_surface.fill((255, 0, 0, flash_alpha))
            self.window.blit(flash_surface, (0, 0))

        # Display status info on top-left with better styling
        cumulative_mining = getattr(self.env, "cumulative_mining_amount", 0.0)
        status_text = [
            f"Energy: {self.env.agent_energy:.0f}/150",
            f"Inventory: {self.env.agent_inventory:.0f}/{self.env.max_inventory}",
            f"Total Mined: {cumulative_mining:.1f}",
            f"Collisions: {self.env.collision_count}",
            f"Step: {self.env.steps_count}/{self.env.max_episode_steps}",
            (
                "Asteroids: "
                f"{np.sum(self.env.asteroid_resources >= 0.1)}/"
                f"{len(self.env.asteroid_positions)}"
            ),
        ]

        if hasattr(self.env, "mining_asteroid_id"):
            status_text.append(f"🔨 MINING: Asteroid {self.env.mining_asteroid_id}")
        elif self.env.agent_inventory > 0:
            status_text.append("📦 CARRYING RESOURCES")
        else:
            status_text.append("🔍 EXPLORING")

        if (
            hasattr(self.env, "last_mining_info")
            and self.env.last_mining_info.get("step", 0) == self.env.steps_count
        ):
            status_text.append(
                f"⛏️ MINED: {self.env.last_mining_info['extracted']:.1f} from Asteroid {self.env.last_mining_info['asteroid_id']}"
            )
            if self.env.last_mining_info.get("asteroid_depleted", False):
                status_text.append("💀 ASTEROID DEPLETED!")

        if (
            hasattr(self.env, "last_delivery_info")
            and self.env.last_delivery_info.get("step", 0) == self.env.steps_count
        ):
            status_text.append(f"🚀 DELIVERED: {self.env.last_delivery_info['delivered']:.1f} resources")
            status_text.append("⚡ FULLY RECHARGED!")

        if (
            hasattr(self.env, "last_collision_step")
            and self.env.last_collision_step == self.env.steps_count
        ):
            status_text.append("💥 COLLISION DETECTED!")

        if hasattr(self.env, "tried_depleted_asteroid") and self.env.tried_depleted_asteroid:
            status_text.append("⚠️ TRIED TO MINE DEPLETED ASTEROID!")

        # Render status text with background
        status_bg = pygame.Surface((300, len(status_text) * 22 + 10), pygame.SRCALPHA)
        status_bg.fill((0, 0, 0, 180))
        self.window.blit(status_bg, (5, 5))
        
        for i, text in enumerate(status_text):
            rendered_text = self.font.render(text, True, (255, 255, 255))
            self.window.blit(rendered_text, (10, 10 + i * 22))

        # Enhanced Legend with better positioning and styling
        legend_text = [
            "LEGEND:",
            "🤖 Green Circle = Agent",
            "🏭 Blue Circle = Mothership", 
            "☄️ Red Circles = Obstacles",
            "🌕 Yellow Circles = Asteroids",
            "💀 Gray X = Depleted Asteroids",
            "🔵 Blue Ring = Observation Range",
            "🔴 Red Ring = Mining Range",
            "⚡ Yellow Beam = Mining Energy",
            "✨ Yellow Dots = Resource Delivery"
        ]
        
        legend_bg = pygame.Surface((280, len(legend_text) * 20 + 10), pygame.SRCALPHA)
        legend_bg.fill((0, 0, 0, 180))
        self.window.blit(legend_bg, (515, 580))
        
        for i, text in enumerate(legend_text):
            color = (255, 255, 255) if i == 0 else (200, 200, 200)
            rendered_text = self.font.render(text, True, color)
            self.window.blit(rendered_text, (520, 585 + i * 20))

        # Update display / timing
        if self.env.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.env.metadata["render_fps"])

        if self.env.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

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
