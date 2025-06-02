import matplotlib.pyplot as plt
import pandas as pd
import pygame

from iql_agents import IQLController
from renderer import Renderer
from team_goal_env_simple import TeamGoalEnvSimple


def smooth(y, window=50):
    """Apply rolling mean smoothing to data"""
    return pd.Series(y).rolling(window, min_periods=1, center=False).mean()


def visualize(env, screen, renderer):
    """Handle pygame events and render environment"""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    
    renderer.render(env)
    pygame.display.flip()
    pygame.time.delay(100)  # Control rendering speed


def main():
    # Environment parameters
    grid_size = 8
    n_agents_per_team = 2
    max_steps = 100
    episodes = 10_000
    smoothing_window = 100
    
    # Create environment and controller
    env = TeamGoalEnvSimple(grid_size=grid_size, n_agents_per_team=n_agents_per_team, max_steps=max_steps)
    controller = IQLController(env)
    
    # Pygame setup for visualization
    pygame.init()
    window_size = 600
    screen = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("MARL IQL Agents")
    font = pygame.font.SysFont("Arial", 20)
    cell_size = (window_size - (grid_size + 1) * 2) // grid_size  # Account for margins
    
    renderer = Renderer(screen, grid_size, cell_size, margin=2, font=font)
    
    # Create render wrapper for training
    def render_wrapper(env):
        visualize(env, screen, renderer)
    
    # Train agents
    print(f"Training {n_agents_per_team * 2} agents on {grid_size}x{grid_size} grid for {episodes} episodes...")
    rewards_team0, rewards_team1, actions_history = controller.train(episodes=episodes, render=render_wrapper)
    
    pygame.quit()
    
    # Plot team reward curves
    plt.figure(figsize=(10, 5))
    plt.plot(smooth(rewards_team0, window=smoothing_window), label="Team 0 Reward", color='blue')
    plt.plot(smooth(rewards_team1, window=smoothing_window), label="Team 1 Reward", color='orange')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"IQL Agents Team Rewards ({grid_size}x{grid_size} grid, {n_agents_per_team * 2} agents)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot action distribution
    actions_per_episode = {0: [], 1: [], 2: [], 3: []}
    for counts in actions_history:
        for action in range(4):
            actions_per_episode[action].append(counts[action])
    
    plt.figure(figsize=(10, 5))
    for action, counts in actions_per_episode.items():
        action_name = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}[action]
        plt.plot(smooth(counts, window=smoothing_window), label=f"{action_name}")
    plt.xlabel("Episode")
    plt.ylabel("Times Action Chosen")
    plt.title("Action Distribution Over Training (All Agents)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()