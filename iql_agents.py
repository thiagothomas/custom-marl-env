import numpy as np
from collections import defaultdict


class IQLAgent:
    """Independent Q-Learning Agent"""
    
    def __init__(self, action_space, observation_size, learning_rate=0.1, discount=0.95,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.action_space = action_space
        self.obs_size = observation_size
        self.lr = learning_rate
        self.gamma = discount
        
        # Exploration parameters
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-table: key = obs tuple, value = action values
        self.q_table = defaultdict(lambda: np.zeros(self.action_space.n))
    
    def get_state_key(self, obs):
        """Convert observation to hashable state key"""
        return tuple(obs.tolist())
    
    def choose_action(self, obs):
        """Epsilon-greedy action selection"""
        state = self.get_state_key(obs)
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit
    
    def update(self, obs, action, reward, next_obs, done):
        """Q-learning update"""
        state = self.get_state_key(obs)
        next_state = self.get_state_key(next_obs)
        
        # Q-learning update rule
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state])
        
        self.q_table[state][action] += self.lr * (target - self.q_table[state][action])
    
    def decay_epsilon(self, episode, max_episodes):
        """Linear epsilon decay"""
        linear_decay = self.epsilon_start - (episode / (max_episodes-1)) * (self.epsilon_start - self.epsilon_end)
        self.epsilon = max(self.epsilon_end, linear_decay)


class IQLController:
    """Controller for multiple IQL agents"""
    
    def __init__(self, env, learning_rate=0.1, discount=0.95,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.env = env
        self.agents = [
            IQLAgent(
                action_space=env.action_space,
                observation_size=env.observation_space.shape[0],
                learning_rate=learning_rate,
                discount=discount,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay
            ) for _ in range(env.n_agents)
        ]
    
    def train(self, episodes=10000, render=None):
        """Train all agents"""
        rewards_team0 = []
        rewards_team1 = []
        actions_history = []
        
        for ep in range(episodes):
            obs, _ = self.env.reset()
            done = False
            episode_rewards = np.zeros(self.env.n_agents)
            action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            
            while not done:
                # Choose actions for all agents
                actions = [agent.choose_action(o) for agent, o in zip(self.agents, obs)]
                
                # Track action distribution
                for a in actions:
                    action_counts[a] += 1
                
                # Environment step
                next_obs, rewards, done, _ = self.env.step(actions)
                
                # Update all agents
                for i, agent in enumerate(self.agents):
                    agent.update(obs[i], actions[i], rewards[i], next_obs[i], done)
                    episode_rewards[i] += rewards[i]
                
                obs = next_obs
                
                # Render last episode if requested
                if render and ep == episodes - 1:
                    render(self.env)
            
            # Decay epsilon for all agents
            for agent in self.agents:
                agent.decay_epsilon(episode=ep, max_episodes=episodes)
            
            # Track team rewards
            n = self.env.n_agents_per_team
            team0_reward = np.sum(episode_rewards[0:n])
            team1_reward = np.sum(episode_rewards[n:2*n])
            
            rewards_team0.append(team0_reward)
            rewards_team1.append(team1_reward)
            actions_history.append(action_counts)
            
            # Progress update
            if ep % 100 == 0:
                print(f"Episode {ep + 1}: Team 0 Reward = {team0_reward:.2f}, "
                      f"Team 1 Reward = {team1_reward:.2f}, Epsilon = {self.agents[0].epsilon:.3f}")
        
        return rewards_team0, rewards_team1, actions_history