# iql_agents.py
import pickle
from collections import defaultdict

import numpy as np


class IQLAgent:
    def __init__(self, action_space, observation_size, learning_rate=0.1, discount=0.99,
                 epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=0.995):
        self.action_space = action_space
        self.obs_size = observation_size
        self.lr = learning_rate
        self.gamma = discount

        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-table: key = obs tuple, value = action values
        self.q_table = defaultdict(lambda: np.zeros(self.action_space.n))

    def get_state_key(self, obs):
        # Discretize obs (already int in your env, so just tuple works)
        return tuple(obs.tolist())

    def choose_action(self, obs):
        state = self.get_state_key(obs)
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update(self, obs, action, reward, next_obs, done):
        state = self.get_state_key(obs)
        next_state = self.get_state_key(next_obs)

        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state])

        self.q_table[state][action] += self.lr * (target - self.q_table[state][action])

    def decay_epsilon(self, episode, max_episodes):
        linear_decay = self.epsilon_start - (episode / (max_episodes-1)) * (self.epsilon_start - self.epsilon_end)
        self.epsilon = max(self.epsilon_end, linear_decay)

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(self.action_space.n), data)


class IQLController:
    def __init__(self, env, learning_rate=0.15, discount=0.95,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.95):
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
        rewards_team0 = []
        rewards_team1 = []
        actions_history = []

        for ep in range(episodes):
            obs, _ = self.env.reset()
            done = False
            episode_rewards = np.zeros(self.env.n_agents)
            action_counts = {0: 0, 1: 0, 2: 0, 3: 0}

            while not done:
                actions = [agent.choose_action(o) for agent, o in zip(self.agents, obs)]
                for a in actions:
                    action_counts[a] += 1

                next_obs, rewards, done, _ = self.env.step(actions)

                for i, agent in enumerate(self.agents):
                    agent.update(obs[i], actions[i], rewards[i], next_obs[i], done)
                    episode_rewards[i] += rewards[i]

                obs = next_obs

                if render and ep == episodes - 1:
                    render(self.env)

            for agent in self.agents:
                agent.decay_epsilon(episode=ep, max_episodes=episodes)

            n = self.env.n_agents_per_team
            team0_reward = np.sum(episode_rewards[0:n])
            team1_reward = np.sum(episode_rewards[n:2 * n])

            rewards_team0.append(team0_reward)
            rewards_team1.append(team1_reward)
            actions_history.append(action_counts)

            if ep % 100 == 0:
                print(
                    f"Episode {ep + 1}: Team 0 Reward = {team0_reward:.2f}, Team 1 Reward = {team1_reward:.2f}, Epsilon = {self.agents[0].epsilon:.3f}")

        return rewards_team0, rewards_team1, actions_history
