# team_goal_env.py

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TeamGoalEnv(gym.Env):
    """
    Simple grid-world MARL env with teams and multiple goals per team.
    - 2 teams (team 0 & 1), 2 agents each (agent IDs 0–3).
    - Multiple goals per team. Agents get +10 when any team member steps on any team goal.
    - Small -0.01 penalty per step to encourage faster completion.
    - Episode ends when all goals are collected or max_steps is reached.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=8, n_agents_per_team=2, max_steps=300, n_goals_per_team=3):
        super().__init__()
        self.grid_size = grid_size
        self.n_agents_per_team = n_agents_per_team
        self.n_teams = 2
        self.n_agents = self.n_agents_per_team * self.n_teams
        self.max_steps = max_steps
        self.n_goals_per_team = n_goals_per_team

        # Observation space
        # [agent_x, agent_y] +
        # [other_agent1_x, other_agent1_y, ... other_agentN_x, other_agentN_y] +
        # [all goals_x, goals_y flattened]
        obs_size = 2 + self.n_teams * self.n_goals_per_team * 2
        low = np.full(obs_size, -1, dtype=np.int32)
        high = np.full(obs_size, self.grid_size - 1, dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        # Action space: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        # Random, non-overlapping agent spawns
        self.agent_positions = []
        taken = set()
        while len(self.agent_positions) < self.n_agents:
            pos = (self.np_random.integers(self.grid_size), self.np_random.integers(self.grid_size))
            if pos not in taken:
                taken.add(pos)
                self.agent_positions.append(list(pos))

        # Random multiple goals, not overlapping agents or each other
        self.team_goals = {team: [] for team in range(self.n_teams)}
        for team in range(self.n_teams):
            while len(self.team_goals[team]) < self.n_goals_per_team:
                pos = (self.np_random.integers(self.grid_size), self.np_random.integers(self.grid_size))
                if pos not in taken:
                    taken.add(pos)
                    self.team_goals[team].append(pos)

        obs = self._get_obs()
        return obs, {}

    def step(self, actions):
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        self.steps += 1

        occupied = {tuple(pos) for pos in self.agent_positions}
        new_positions = [list(pos) for pos in self.agent_positions]

        # ----- Track distances before movement -----
        old_distances = []
        for i, pos in enumerate(self.agent_positions):
            team = i // self.n_agents_per_team
            distances = [abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]) for goal in self.team_goals[team]]
            old_distances.append(min(distances) if distances else 0)

        # ----- Move agents -----
        move_dict = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        wall_penalty = -1.0
        collision_penalty = -2.0
        idle_penalty = -0.1
        time_penalty = -0.05

        # (optional) track previous positions for dwell penalty
        if not hasattr(self, 'prev_positions'):
            self.prev_positions = [list(pos) for pos in self.agent_positions]

        for i, act in enumerate(actions):
            x, y = self.agent_positions[i]
            dx, dy = move_dict.get(act, (0, 0))
            nx, ny = x + dx, y + dy

            # Wall check
            if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
                rewards[i] += wall_penalty
                nx, ny = x, y
            elif (nx, ny) not in occupied:
                occupied.remove((x, y))
                occupied.add((nx, ny))
                new_positions[i] = [nx, ny]
            else:
                rewards[i] += collision_penalty
                nx, ny = x, y

            # Idle penalty if stayed still
            if (nx, ny) == (x, y):
                rewards[i] += idle_penalty

        # ----- Update positions -----
        self.agent_positions = new_positions

        # ----- Distance reward shaping -----
        shaping_factor = 1.0
        for i, pos in enumerate(self.agent_positions):
            team = i // self.n_agents_per_team
            distances = [abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]) for goal in self.team_goals[team]]
            new_dist = min(distances) if distances else 0

            # ✅ disable shaping if near any goal
            near_goal = any(abs(pos[0] - g[0]) + abs(pos[1] - g[1]) <= 1 for g in self.team_goals[team])

            if not near_goal and old_distances[i] > 0 and new_dist >= 0:
                reward_shaping = (old_distances[i] ** 1.2 - new_dist ** 1.2) * shaping_factor
                rewards[i] += reward_shaping

            # ✅ dwell penalty (same position as last step)
            if self.prev_positions[i] == self.agent_positions[i]:
                rewards[i] -= 0.1

            # ✅ time penalty every step
            rewards[i] += time_penalty

        # Save positions for next step
        self.prev_positions = [list(pos) for pos in self.agent_positions]

        # ----- Goal collection -----
        goal_reward = 10.0
        for team in range(self.n_teams):
            team_agents = range(team * self.n_agents_per_team, (team + 1) * self.n_agents_per_team)
            goals_to_remove = []
            for goal_pos in self.team_goals[team]:
                for agent in team_agents:
                    if tuple(self.agent_positions[agent]) == goal_pos:
                        rewards[agent] += goal_reward
                        goals_to_remove.append(goal_pos)
                        break
            for goal in goals_to_remove:
                self.team_goals[team].remove(goal)

        done = (self.steps >= self.max_steps) or all(len(goals) == 0 for goals in self.team_goals.values())
        obs = self._get_obs()
        return obs, rewards.tolist(), done, {}

    def _get_obs(self):
        obs = []
        total_goal_slots = self.n_teams * self.n_goals_per_team

        for i, pos in enumerate(self.agent_positions):
            agent_obs = []

            # own pos
            agent_obs.extend(pos)

            # # other agents
            # for j, other_pos in enumerate(self.agent_positions):
            #     if i != j:
            #         agent_obs.extend(other_pos)

            # goals
            goals = []
            for team_goals in self.team_goals.values():
                for goal_pos in team_goals:
                    goals.extend(goal_pos)
            while len(goals) < total_goal_slots * 2:
                goals.extend([-1, -1])

            agent_obs.extend(goals)
            obs.append(np.array(agent_obs, dtype=np.int32))

        return obs
