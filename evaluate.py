import numpy as np
import pickle
import pygame
from collections import defaultdict

from team_goal_env_simple import TeamGoalEnvSimple
from renderer import Renderer


class EvaluationAgent:
    """Agent that uses a pre-trained Q-table for evaluation"""
    
    def __init__(self, q_table, action_space):
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))
        # Load the saved Q-table
        for state, values in q_table.items():
            self.q_table[state] = values
        self.action_space = action_space
    
    def get_state_key(self, obs):
        """Convert observation to hashable state key"""
        return tuple(obs.tolist())
    
    def choose_action(self, obs):
        """Choose action greedily from Q-table (no exploration)"""
        state = self.get_state_key(obs)
        return np.argmax(self.q_table[state])


def visualize(env, screen, renderer):
    """Handle pygame events and render environment"""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    
    renderer.render(env)
    pygame.display.flip()
    pygame.time.delay(200)  # Slower for evaluation


def evaluate_agents(episodes=10, render=True):
    """Evaluate saved agents using their Q-tables"""
    
    # Load saved Q-tables and environment parameters
    print("Loading saved Q-tables...")
    
    with open("saved_models/team0_qtables.pkl", "rb") as f:
        team0_qtables = pickle.load(f)
    
    with open("saved_models/team1_qtables.pkl", "rb") as f:
        team1_qtables = pickle.load(f)
    
    with open("saved_models/env_params.pkl", "rb") as f:
        env_params = pickle.load(f)
    
    print(f"Loaded Q-tables for {len(team0_qtables)} agents in Team 0 and {len(team1_qtables)} agents in Team 1")
    
    # Create environment with same parameters
    env = TeamGoalEnvSimple(
        grid_size=env_params["grid_size"],
        n_agents_per_team=env_params["n_agents_per_team"],
        max_steps=env_params["max_steps"]
    )
    
    # Create evaluation agents
    agents = []
    
    # Team 0 agents
    for q_table in team0_qtables:
        agents.append(EvaluationAgent(q_table, env.action_space))
    
    # Team 1 agents
    for q_table in team1_qtables:
        agents.append(EvaluationAgent(q_table, env.action_space))
    
    # Pygame setup if rendering
    if render:
        pygame.init()
        window_size = 600
        screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("MARL IQL Agents - Evaluation")
        font = pygame.font.SysFont("Arial", 20)
        cell_size = (window_size - (env_params["grid_size"] + 1) * 2) // env_params["grid_size"]
        renderer = Renderer(screen, env_params["grid_size"], cell_size, margin=2, font=font)
    
    # Run evaluation episodes
    total_rewards_team0 = []
    total_rewards_team1 = []
    goals_collected_team0 = []
    goals_collected_team1 = []
    
    print(f"\nRunning {episodes} evaluation episodes...")
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_rewards = np.zeros(env.n_agents)
        steps = 0
        
        # Track initial goals
        initial_goals_team0 = len(env.team_goals[0])
        initial_goals_team1 = len(env.team_goals[1])
        
        while not done:
            # Choose actions using optimal policy (greedy)
            actions = [agent.choose_action(o) for agent, o in zip(agents, obs)]
            
            # Environment step
            next_obs, rewards, done, _ = env.step(actions)
            
            # Accumulate rewards
            for i in range(env.n_agents):
                episode_rewards[i] += rewards[i]
            
            obs = next_obs
            steps += 1
            
            # Render if requested
            if render:
                visualize(env, screen, renderer)
        
        # Calculate team rewards and goals collected
        n = env.n_agents_per_team
        team0_reward = np.sum(episode_rewards[0:n])
        team1_reward = np.sum(episode_rewards[n:2*n])
        
        # Count remaining goals
        final_goals_team0 = len(env.team_goals[0])
        final_goals_team1 = len(env.team_goals[1])
        
        # Calculate goals collected
        goals_team0 = initial_goals_team0 - final_goals_team0
        goals_team1 = initial_goals_team1 - final_goals_team1
        
        total_rewards_team0.append(team0_reward)
        total_rewards_team1.append(team1_reward)
        goals_collected_team0.append(goals_team0)
        goals_collected_team1.append(goals_team1)
        
        print(f"Episode {ep + 1}: Team 0 Reward = {team0_reward:.2f} (Goals: {goals_team0}), "
              f"Team 1 Reward = {team1_reward:.2f} (Goals: {goals_team1}), Steps = {steps}")
    
    if render:
        pygame.quit()
    
    # Print evaluation summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Team 0 - Average Reward: {np.mean(total_rewards_team0):.2f} ± {np.std(total_rewards_team0):.2f}")
    print(f"Team 0 - Average Goals: {np.mean(goals_collected_team0):.2f} ± {np.std(goals_collected_team0):.2f}")
    print(f"Team 1 - Average Reward: {np.mean(total_rewards_team1):.2f} ± {np.std(total_rewards_team1):.2f}")
    print(f"Team 1 - Average Goals: {np.mean(goals_collected_team1):.2f} ± {np.std(goals_collected_team1):.2f}")
    print("="*50)


if __name__ == "__main__":
    # Run evaluation with rendering
    evaluate_agents(episodes=10, render=True)