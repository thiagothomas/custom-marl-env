# renderer.py

import pygame
from team_goal_env import TeamGoalEnv

COLORS = {
    "background": (30, 30, 30),
    "grid": (50, 50, 50),
    "team0": (0, 100, 255),
    "team1": (255, 100, 0),
}


class Renderer:
    def __init__(self, screen, grid_size, cell_size, margin, font):
        self.screen = screen
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.margin = margin
        self.font = font

    def render(self, env: TeamGoalEnv):
        # Draw background grid
        self.screen.fill(COLORS["background"])
        for y in range(env.grid_size):
            for x in range(env.grid_size):
                rect = pygame.Rect(
                    self.margin + x * (self.cell_size + self.margin),
                    self.margin + y * (self.cell_size + self.margin),
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, COLORS["grid"], rect)

        # Draw ALL goals for each team
        for team, goals in env.team_goals.items():
            for gx, gy in goals:
                if 0 <= gx < env.grid_size and 0 <= gy < env.grid_size:
                    rect = pygame.Rect(
                        self.margin + gx * (self.cell_size + self.margin),
                        self.margin + gy * (self.cell_size + self.margin),
                        self.cell_size,
                        self.cell_size
                    )
                    pygame.draw.rect(self.screen, COLORS[f"team{team}"], rect)

        # Draw agents
        for idx, (ax, ay) in enumerate(env.agent_positions):
            team = idx // env.n_agents_per_team
            color = COLORS[f"team{team}"]
            center = (
                self.margin + ax * (self.cell_size + self.margin) + self.cell_size // 2,
                self.margin + ay * (self.cell_size + self.margin) + self.cell_size // 2
            )
            radius = self.cell_size // 3
            pygame.draw.circle(self.screen, color, center, radius)

            # Label agent
            member = idx % env.n_agents_per_team
            label = chr(ord('a') + member) if team == 0 else chr(ord('A') + member)
            text = self.font.render(label, True, (255, 255, 255))
            self.screen.blit(text, text.get_rect(center=center))
