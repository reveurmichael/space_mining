from typing import Any, Optional

import numpy as np


class Renderer:
    """Minimal, clean renderer for the SpaceMining environment.

    Focuses on:
    - Clear scene: starfield, mothership, asteroids, obstacles, agent
    - Simple overlays: ranges, energy bar, concise status panel
    - No toggles/interactive UI; optimized for GIF clarity
    """

    def __init__(self, env: Any) -> None:
        self.env: Any = env
        self.window: Optional[Any] = None
        self.clock: Optional[Any] = None
        self.font: Optional[Any] = None

        # Simple background starfield
        self.starfield = []
        self._init_starfield()

    def _init_starfield(self) -> None:
        """Create a simple, subtle starfield for 1920x1080."""
        import numpy as np
        self.starfield = []
        for _ in range(500):
            self.starfield.append({
                "x": float(np.random.uniform(0, 1920)),
                "y": float(np.random.uniform(0, 1080)),
                "size": int(np.random.choice([1, 2, 3], p=[0.7, 0.25, 0.05])),
                "brightness": int(np.random.randint(20, 100)),
                "drift_x": float(np.random.uniform(-0.03, 0.03)),
                "drift_y": float(np.random.uniform(-0.03, 0.03)),
                "color_type": str(np.random.choice(["white", "blue", "yellow"], p=[0.85, 0.1, 0.05]))
            })

    def _update_starfield(self) -> None:
        """Update starfield with gentle drift and wrap-around."""
        for star in self.starfield:
            star["x"] += star.get("drift_x", 0.0)
            star["y"] += star.get("drift_y", 0.0)
            if star["x"] < -5:
                star["x"] = 1925
            elif star["x"] > 1925:
                star["x"] = -5
            if star["y"] < -5:
                star["y"] = 1085
            elif star["y"] > 1085:
                star["y"] = -5

    def render(self) -> Optional[np.ndarray]:
        """Render the current state of the environment."""
        if self.env.render_mode is None:
            return

        try:
            import pygame
            from pygame import gfxdraw
            import math
        except ImportError as exc:
            raise ImportError("pygame is not installed, run `pip install pygame`") from exc

        if self.window is None:
            pygame.init()
            if self.env.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode((1920, 1080))
                pygame.display.set_caption("Space Mining")
            else:
                self.window = pygame.Surface((1920, 1080))
            if self.clock is None:
                self.clock = pygame.time.Clock()
            if self.font is None:
                pygame.font.init()
                self.font = pygame.font.SysFont("Arial", 18)

        # Update background and animations
        self._update_starfield()
        self.update_animations()
        self.update_zoom()

        # Clear background
        self.window.fill((0, 0, 3))

        # Draw starfield
        self._draw_starfield()

        # Coordinate transformation (1920x1080 center with zoom)
        def to_screen(pos, scale=10.0):
            x, y = pos
            zoom_scale = scale * self.env.zoom_level
            screen_x = int(960 + (x - 40) * zoom_scale)
            screen_y = int(540 + (y - 40) * zoom_scale)
            return screen_x, screen_y

        # Mothership
        mothership_pos_2d = to_screen(self.env.mothership_pos)
        gfxdraw.filled_circle(self.window, mothership_pos_2d[0], mothership_pos_2d[1], 18, (30, 120, 200))
        gfxdraw.aacircle(self.window, mothership_pos_2d[0], mothership_pos_2d[1], 18, (200, 220, 255))

        # Asteroids
        for i, pos in enumerate(self.env.asteroid_positions):
            asteroid_pos_2d = to_screen(pos)
            if self.env.asteroid_resources[i] < 0.1:
                gfxdraw.filled_circle(self.window, asteroid_pos_2d[0], asteroid_pos_2d[1], 8, (80, 80, 80))
                pygame.draw.line(self.window, (150, 150, 150), (asteroid_pos_2d[0]-6, asteroid_pos_2d[1]-6), (asteroid_pos_2d[0]+6, asteroid_pos_2d[1]+6), 3)
                pygame.draw.line(self.window, (150, 150, 150), (asteroid_pos_2d[0]+6, asteroid_pos_2d[1]-6), (asteroid_pos_2d[0]-6, asteroid_pos_2d[1]+6), 3)
                continue
            resource_ratio = self.env.asteroid_resources[i] / 40.0
            size = int(10 + resource_ratio * 10)
            gfxdraw.filled_circle(self.window, asteroid_pos_2d[0], asteroid_pos_2d[1], size, (255, 215, 0))
            # Health bar
            health_ratio = resource_ratio
            health_width = int(28 * health_ratio)
            health_x = asteroid_pos_2d[0] - 14
            health_y = asteroid_pos_2d[1] - size - 12
            pygame.draw.rect(self.window, (40, 40, 40), (health_x, health_y, 28, 6))
            if health_ratio > 0.6:
                health_color = (0, 255, 0)
            elif health_ratio > 0.3:
                health_color = (255, 255, 0)
            else:
                health_color = (255, 100, 0)
            pygame.draw.rect(self.window, health_color, (health_x, health_y, health_width, 6))

        # Obstacles
        for pos in self.env.obstacle_positions:
            obstacle_pos_2d = to_screen(pos)
            gfxdraw.filled_circle(self.window, obstacle_pos_2d[0], obstacle_pos_2d[1], 10, (220, 50, 50))
            gfxdraw.aacircle(self.window, obstacle_pos_2d[0], obstacle_pos_2d[1], 12, (255, 120, 120))

        # Agent
        agent_pos_2d = to_screen(self.env.agent_position)
        gfxdraw.filled_circle(self.window, agent_pos_2d[0], agent_pos_2d[1], 15, (50, 255, 50))
        gfxdraw.aacircle(self.window, agent_pos_2d[0], agent_pos_2d[1], 15, (255, 255, 255))

        # Energy warning halo
        if self.env.agent_energy < 30:
            pygame.draw.circle(self.window, (255, 0, 0), (agent_pos_2d[0], agent_pos_2d[1]), 20, 2)

        # Mining beam
        if hasattr(self.env, "mining_asteroid_id") and self.env.mining_asteroid_id is not None:
            asteroid_pos_2d = to_screen(self.env.asteroid_positions[self.env.mining_asteroid_id])
            pygame.draw.line(self.window, (255, 200, 50), agent_pos_2d, asteroid_pos_2d, 4)

        # Energy bar
        energy_ratio = self.env.agent_energy / 150.0
        energy_width = int(70 * energy_ratio)
        bar_x = agent_pos_2d[0] - 35
        bar_y = agent_pos_2d[1] - 35
        pygame.draw.rect(self.window, (20, 20, 20), (bar_x - 2, bar_y - 2, 74, 12))
        pygame.draw.rect(self.window, (60, 60, 60), (bar_x, bar_y, 70, 8))
        if energy_ratio > 0.6:
            energy_color = (0, 255, 0)
        elif energy_ratio > 0.3:
            energy_color = (255, 255, 0)
        else:
            energy_color = (255, 50, 50)
        pygame.draw.rect(self.window, energy_color, (bar_x, bar_y, energy_width, 8))

        # Ranges
        obs_radius_px = int(self.env.observation_radius * 14.0 * self.env.zoom_level)
        mining_radius_px = int(self.env.mining_range * 14.0 * self.env.zoom_level)
        gfxdraw.aacircle(self.window, agent_pos_2d[0], agent_pos_2d[1], obs_radius_px, (100, 150, 255))
        gfxdraw.aacircle(self.window, agent_pos_2d[0], agent_pos_2d[1], mining_radius_px, (255, 100, 100))

        # Collision flash overlay
        if self.env.collision_flash_timer > 0:
            alpha = int(140 * (self.env.collision_flash_timer / 0.3))
            overlay = pygame.Surface((1920, 1080), pygame.SRCALPHA)
            overlay.fill((255, 0, 0, alpha))
            self.window.blit(overlay, (0, 0))

        # Status panel
        self._draw_status_panel()

        # Game over screen
        if self.env.game_over_state.get("active", False):
            self._draw_game_over_screen()

        # Update display/timing
        if self.env.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.env.metadata["render_fps"])

        if self.env.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2))

    def _draw_starfield(self) -> None:
        try:
            from pygame import gfxdraw
        except ImportError:
            return
        for star in self.starfield:
            if not (0 <= star["x"] <= 1920 and 0 <= star["y"] <= 1080):
                continue
            x, y = int(star["x"]), int(star["y"])
            size = star["size"]
            brightness = star["brightness"]
            if star["color_type"] == "blue":
                color = (brightness//5, brightness//2, brightness)
            elif star["color_type"] == "yellow":
                color = (brightness, int(brightness*0.7), brightness//5)
            else:
                color = (brightness, brightness, brightness)
            if size == 1:
                gfxdraw.pixel(self.window, x, y, color)
            else:
                gfxdraw.filled_circle(self.window, x, y, size-1, color)

    def _draw_status_panel(self) -> None:
        try:
            import pygame
            import numpy as np
            import math
        except ImportError:
            return

        # Metrics
        fitness = self.env.compute_fitness_score()
        vx, vy = float(self.env.agent_velocity[0]), float(self.env.agent_velocity[1])
        speed = float(np.linalg.norm(self.env.agent_velocity))
        heading = (math.degrees(math.atan2(vy, vx)) % 360) if speed > 1e-4 else None
        dist_to_mothership = float(np.linalg.norm(self.env.agent_position - self.env.mothership_pos))

        # State/mining
        if hasattr(self.env, "mining_asteroid_id") and self.env.mining_asteroid_id is not None:
            state_label = f"Mining A{self.env.mining_asteroid_id}"
            mining_status = f"Active (A{self.env.mining_asteroid_id})"
            state_color = (255, 255, 100)
        elif self.env.agent_inventory > 0:
            state_label = f"Carrying {self.env.agent_inventory:.0f}"
            mining_status = "Not mining"
            state_color = (255, 255, 0)
        else:
            state_label = "Searching"
            mining_status = "Not mining"
            state_color = (200, 200, 255)

        energy_pct = int(self.env.agent_energy / 150.0 * 100)
        inv = self.env.agent_inventory
        inv_max = getattr(self.env, "max_inventory", 100)

        items = [
            ("State:", state_label, state_color),
            ("Step Count:", f"{self.env.steps_count} of {self.env.max_episode_steps}", (200, 200, 255)),
            ("Energy Level:", f"{int(self.env.agent_energy)}/150 ({energy_pct}%)", (255, 100, 100) if energy_pct < 30 else (100, 255, 100)),
            ("Inventory:", f"{inv:.1f} of {inv_max}", (255, 255, 100) if inv > 0 else (180, 180, 180)),
            ("Mining:", mining_status, (255, 255, 100)),
            ("Deliveries:", f"{getattr(self.env, 'delivery_count', 0)}", (0, 255, 0)),
            ("Fitness Score:", f"{fitness:.1f}", (255, 215, 100)),
            ("Speed:", f"{speed:.2f}", (180, 180, 255)),
            ("Speed X:", f"{vx:.2f}", (180, 180, 255)),
            ("Speed Y:", f"{vy:.2f}", (180, 180, 255)),
            ("Heading (deg):", f"{heading:.0f}" if heading is not None else "N/A", (180, 180, 255)),
            ("Distance to Mothership:", f"{dist_to_mothership:.1f}", (30, 120, 200)),
            ("Collisions:", f"{self.env.collision_count}", (255, 100, 100) if self.env.collision_count > 0 else (200, 200, 200)),
            ("Action:", f"({self.env.last_action[0]:.2f}, {self.env.last_action[1]:.2f}, {'Mining' if self.env.last_action[2]>0.5 else 'Not Mining'})", (200, 200, 200)),
        ]

        # Panel geometry
        item_width = 320
        item_height = 32
        padding = 12
        panel_width = item_width + padding * 2
        title_height = 30
        items_height = len(items) * item_height
        panel_height = title_height + items_height + 16

        # Background
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 200))
        pygame.draw.rect(panel, (60, 60, 80), (0, 0, panel_width, panel_height), 2)

        # Title
        title_font = pygame.font.SysFont("Arial", 18, bold=True)
        title_surface = title_font.render("STATUS", True, state_color)
        title_rect = title_surface.get_rect(center=(panel_width // 2, 18))
        panel.blit(title_surface, title_rect)

        # Items
        label_font = pygame.font.SysFont("Arial", 15)
        value_font = pygame.font.SysFont("Arial", 15, bold=False)
        for i, (label, value, color) in enumerate(items):
            y = 32 + i * item_height
            label_surface = label_font.render(label, True, (200, 200, 210))
            value_surface = value_font.render(str(value), True, color)
            panel.blit(label_surface, (padding, y))
            panel.blit(value_surface, (padding + 130, y))

        # Blit
        self.window.blit(panel, (15, 15))

    def _draw_game_over_screen(self) -> None:
        try:
            import pygame
        except ImportError:
            return
        if self.env.game_over_state.get("fade_alpha", 0) < 255:
            self.env.game_over_state["fade_alpha"] = self.env.game_over_state.get("fade_alpha", 0) + 3
        fade_alpha = min(255, self.env.game_over_state.get("fade_alpha", 255))
        fade_surface = pygame.Surface((1920, 1080), pygame.SRCALPHA)
        fade_surface.fill((0, 0, 0, fade_alpha))
        self.window.blit(fade_surface, (0, 0))

        if fade_alpha > 100:
            stats = self.env.game_over_state.get("final_stats", {})
            success = self.env.game_over_state.get("success", False)
            title_font = pygame.font.SysFont("Arial", 56, bold=True)
            title_text = "MISSION SUCCESS" if success else "MISSION FAILED"
            title_color = (0, 255, 0) if success else (255, 100, 100)
            title_surface = title_font.render(title_text, True, title_color)
            title_rect = title_surface.get_rect(center=(960, 280))
            self.window.blit(title_surface, title_rect)

            stats_font = pygame.font.SysFont("Arial", 24)
            lines = [
                f"Total Mined: {stats.get('total_resources_mined', 0.0):.1f}",
                f"Delivered: {stats.get('resources_delivered', 0.0):.1f}",
                f"Inventory: {stats.get('current_inventory', 0.0):.1f}",
                f"Collisions: {stats.get('collisions', 0)}",
                f"Steps: {stats.get('steps_taken', 0)}",
                f"Energy: {stats.get('final_energy', 0.0):.1f}/150",
                f"Asteroids Depleted: {stats.get('asteroids_depleted', 0)}/{stats.get('total_asteroids', 0)}",
                f"Efficiency: {stats.get('efficiency_score', 0.0):.0f}",
            ]
            y = 380
            for line in lines:
                ts = stats_font.render(line, True, (230, 230, 230))
                tr = ts.get_rect(center=(960, y))
                self.window.blit(ts, tr)
                y += 36

    def update_zoom(self) -> None:
        """Update zoom with simple, readable behavior."""
        zoom_diff = self.env.target_zoom - self.env.zoom_level
        self.env.zoom_speed = 0.025
        self.env.zoom_level += zoom_diff * self.env.zoom_speed
        if getattr(self.env, 'agent_energy', 999) < 20:
            self.env.target_zoom = 1.2
        elif getattr(self.env, 'collision_flash_timer', 0) > 0:
            self.env.target_zoom = 0.9
        elif len([a for a in self.env.asteroid_resources if a > 0.1]) <= 2:
            self.env.target_zoom = 0.95
        elif getattr(self.env, 'mining_asteroid_id', None) is not None:
            self.env.target_zoom = 1.05
        else:
            self.env.target_zoom = 1.0
        self.env.zoom_level = max(0.85, min(1.25, self.env.zoom_level))

    def spawn_delivery_particles(self, start_pos: np.ndarray, target_pos: np.ndarray) -> None:
        for _ in range(10):
            particle = {"start_pos": start_pos.copy(), "target_pos": target_pos.copy(), "progress": 0.0}
            self.env.delivery_particles.append(particle)

    def add_score_popup(self, text: str, pos: np.ndarray, color: tuple) -> None:
        popup = {"text": text, "pos": pos.copy(), "alpha": 255, "color": color}
        self.env.score_popups.append(popup)

    def update_animations(self) -> None:
        for particle in self.env.delivery_particles:
            particle["progress"] += 0.05
            if particle["progress"] >= 1.0:
                particle["progress"] = 1.0
        self.env.delivery_particles = [p for p in self.env.delivery_particles if p["progress"] < 1.0]

        for popup in self.env.score_popups:
            popup["pos"][1] -= 0.3
            popup["alpha"] -= 5
        self.env.score_popups = [p for p in self.env.score_popups if p["alpha"] > 0]

        if self.env.collision_flash_timer > 0:
            self.env.collision_flash_timer -= self.env.dt
        if self.env.screen_shake_timer > 0:
            self.env.screen_shake_timer -= self.env.dt
        self.env.mining_beam_offset += 0.2

    def trigger_game_over(self, success: bool) -> None:
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
                "efficiency_score": self.env.compute_fitness_score(),
            },
        }

    def close(self) -> None:
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

