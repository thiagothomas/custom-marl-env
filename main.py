import matplotlib.pyplot as plt
import pandas as pd
import pygame

from iql_agents import IQLController
from renderer import Renderer
from team_goal_env import TeamGoalEnv


def smooth(y, window=50):
    return pd.Series(y).rolling(window, min_periods=1, center=False).mean()


def visualize(env, screen, renderer):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    renderer.render(env)
    pygame.display.flip()
    pygame.time.delay(100)  # speed of rendering


def main():
    # --- setup ---
    grid_size = 5
    n_agents_per_team = 1
    max_steps = 50
    episodes = 1_000_000
    smoothing_window = 75

    env = TeamGoalEnv(grid_size=grid_size, n_agents_per_team=n_agents_per_team, max_steps=max_steps)
    controller = IQLController(env)
    # controller = TeamQController(env)

    # --- pygame setup ---
    pygame.init()
    window_size = 600
    screen = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("MARL IQL Agents")
    font = pygame.font.SysFont("Arial", 20)
    cell_size = (window_size - (grid_size + 1) * 2) // grid_size  # 2px margin

    renderer = Renderer(screen, grid_size, cell_size, margin=2, font=font)

    # --- train ---
    def render_wrapper(env):
        visualize(env, screen, renderer)

    rewards_team0, rewards_team1, actions_history = controller.train(episodes=episodes, render=render_wrapper)

    pygame.quit()

    # --- plot team reward curves (smoothed) ---
    plt.figure(figsize=(10, 5))
    plt.plot(smooth(rewards_team0, window=smoothing_window), label="Team 0 Reward", color='blue')
    plt.plot(smooth(rewards_team1, window=smoothing_window), label="Team 1 Reward", color='orange')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"IQL Agents Team Rewards ({grid_size}x{grid_size} grid, {n_agents_per_team * 2} agents)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- plot action distribution (smoothed) ---
    actions_per_episode = {0: [], 1: [], 2: [], 3: []}
    for counts in actions_history:
        for action in range(4):
            actions_per_episode[action].append(counts[action])

    plt.figure(figsize=(10, 5))
    for action, counts in actions_per_episode.items():
        action_name = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}[action]
        plt.plot(smooth(counts, window=smoothing_window), label=f"{action_name} ({action})")
    plt.xlabel("Episode")
    plt.ylabel("Times action was chosen")
    plt.title("Actions chosen per episode (all agents combined)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
