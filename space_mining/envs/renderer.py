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

        self.window.fill((0, 0, 0))  # Space background (black)

        # Helper function to convert 2D coordinates to screen coordinates
        def to_screen(pos, scale=8.0):
            x, y = pos
            # Center the 80x80 grid in the 800x800 screen
            # Map 0-80 to 0-800, centered at 400
            screen_x = int(
                400 + (x - 40) * scale
            )  # Center at 400, scale from -40 to +40
            screen_y = int(
                400 + (y - 40) * scale
            )  # Center at 400, scale from -40 to +40
            return screen_x, screen_y

        # Draw mothership
        mothership_pos_2d = to_screen(self.env.mothership_pos)
        gfxdraw.filled_circle(
            self.window, mothership_pos_2d[0], mothership_pos_2d[1], 15, (50, 150, 200)
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
            size = int(5 + self.env.asteroid_resources[i] / 10)
            color_intensity = min(255, int(100 + self.env.asteroid_resources[i] * 3))
            gfxdraw.filled_circle(
                self.window,
                asteroid_pos_2d[0],
                asteroid_pos_2d[1],
                size,
                (color_intensity, color_intensity // 2, 0),
            )

            # Draw health bar for asteroid
            health_ratio = (
                self.env.asteroid_resources[i] / 40.0
            )  # Normalize to max resource
            health_width = int(20 * health_ratio)
            health_x = asteroid_pos_2d[0] - 10
            health_y = asteroid_pos_2d[1] - size - 8

            # Background (gray)
            pygame.draw.rect(self.window, (100, 100, 100), (health_x, health_y, 20, 4))
            # Health bar (green to red based on remaining resources)
            health_color = (
                int(255 * health_ratio),
                int(255 * (1 - health_ratio)),
                0,
            )  # Green to Red
            pygame.draw.rect(
                self.window, health_color, (health_x, health_y, health_width, 4)
            )

        # Draw obstacles
        for pos in self.env.obstacle_positions:
            obstacle_pos_2d = to_screen(pos)
            gfxdraw.filled_circle(
                self.window, obstacle_pos_2d[0], obstacle_pos_2d[1], 7, (200, 50, 50)
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

        # Draw mining indicator
        if hasattr(self.env, "mining_asteroid_id"):
            asteroid_pos_2d = to_screen(
                self.env.asteroid_positions[self.env.mining_asteroid_id]
            )
            pygame.draw.line(
                self.window,
                (255, 255, 0),  # Yellow beam
                agent_pos_2d,
                asteroid_pos_2d,
                3,
            )
            mining_text = f"MINING ASTEROID {self.env.mining_asteroid_id}"
            text_surface = self.font.render(mining_text, True, (255, 255, 0))
            self.window.blit(text_surface, (agent_pos_2d[0] - 50, agent_pos_2d[1] - 30))

        # Draw inventory indicator
        if self.env.agent_inventory > 0:
            gfxdraw.filled_circle(
                self.window,
                agent_pos_2d[0],
                agent_pos_2d[1],
                int(5 + self.env.agent_inventory / 5),
                (200, 200, 0),
            )

        # Draw energy bar
        energy_width = int(40 * (self.env.agent_energy / 150.0))
        pygame.draw.rect(
            self.window, (0, 0, 0), (agent_pos_2d[0] - 20, agent_pos_2d[1] - 20, 40, 5)
        )
        pygame.draw.rect(
            self.window,
            (0, 200, 0),
            (agent_pos_2d[0] - 20, agent_pos_2d[1] - 20, energy_width, 5),
        )

        # Draw observation radius
        gfxdraw.aacircle(
            self.window,
            agent_pos_2d[0],
            agent_pos_2d[1],
            int(self.env.observation_radius * 8.0),
            (100, 100, 255),
        )

        # Draw mining range
        gfxdraw.aacircle(
            self.window,
            agent_pos_2d[0],
            agent_pos_2d[1],
            int(self.env.mining_range * 8.0),
            (255, 0, 0),
        )

        # Draw mining range indicator text
        mining_text = f"MINING RANGE: {self.env.mining_range:.1f}"
        text_surface = self.font.render(mining_text, True, (255, 0, 0))
        self.window.blit(text_surface, (agent_pos_2d[0] - 50, agent_pos_2d[1] + 30))

        # Display status info on top-left
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
            status_text.append(f"MINING: Asteroid {self.env.mining_asteroid_id}")
        elif self.env.agent_inventory > 0:
            status_text.append("CARRYING RESOURCES")
        else:
            status_text.append("EXPLORING")

        if (
            hasattr(self.env, "last_mining_info")
            and self.env.last_mining_info.get("step", 0) == self.env.steps_count
        ):
            status_text.append(
                (
                    "MINED: "
                    f"{self.env.last_mining_info['extracted']:.1f} from Asteroid "
                    f"{self.env.last_mining_info['asteroid_id']}"
                )
            )
            status_text.append(
                (
                    "TOTAL MINED: "
                    f"{self.env.last_mining_info.get('cumulative_mining', 0.0):.1f}"
                )
            )
            if self.env.last_mining_info.get("asteroid_depleted", False):
                status_text.append("ASTEROID DEPLETED!")

        if (
            hasattr(self.env, "last_delivery_info")
            and self.env.last_delivery_info.get("step", 0) == self.env.steps_count
        ):
            status_text.append(
                f"DELIVERED: {self.env.last_delivery_info['delivered']:.1f} resources"
            )
            status_text.append("FULLY RECHARGED!")

        if (
            hasattr(self.env, "last_collision_step")
            and self.env.last_collision_step == self.env.steps_count
        ):
            status_text.append("COLLISION DETECTED!")

        if hasattr(self.env, "tried_depleted_asteroid") and self.env.tried_depleted_asteroid:
            status_text.append("TRIED TO MINE DEPLETED ASTEROID!")

        # Render status text
        for i, text in enumerate(status_text):
            rendered_text = self.font.render(text, True, (255, 255, 255))
            self.window.blit(rendered_text, (10, 10 + i * 20))

        # Legend
        legend_text = [
            "LEGEND:",
            "Green Circle = Agent",
            "Blue Circle = Mothership",
            "Red Circles = Obstacles",
            "Yellow Circles = Asteroids",
            "Gray X = Depleted Asteroids",
            "Blue Ring = Observation Range",
            "Red Ring = Mining Range",
        ]
        for i, text in enumerate(legend_text):
            rendered_text = self.font.render(text, True, (255, 255, 255))
            self.window.blit(rendered_text, (600, 600 + i * 20))

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
