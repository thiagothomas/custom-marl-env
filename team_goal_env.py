import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TeamGoalEnv(gym.Env):
    """
    Multi-agent grid world environment with team-based goal collection.
    
    Features:
    - 2 teams with configurable agents per team
    - Multiple goals per team scattered on the grid
    - Agents receive rewards for collecting their team's goals
    - Distance-based reward shaping to guide agents
    - Penalties for collisions and time
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
        
        # Observation space: [agent_x, agent_y, team_id, goal_positions...]
        obs_size = 3 + self.n_teams * self.n_goals_per_team * 2
        low = np.full(obs_size, -1, dtype=np.int32)
        high = np.full(obs_size, self.grid_size - 1, dtype=np.int32)
        low[2] = 0  # team_id minimum
        high[2] = self.n_teams - 1  # team_id maximum
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)
        
        # Action space: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        
        # Initialize agent positions randomly without overlap
        self.agent_positions = []
        taken = set()
        while len(self.agent_positions) < self.n_agents:
            pos = (self.np_random.integers(self.grid_size), self.np_random.integers(self.grid_size))
            if pos not in taken:
                taken.add(pos)
                self.agent_positions.append(list(pos))
        
        # Initialize team goals randomly without overlap
        self.team_goals = {team: [] for team in range(self.n_teams)}
        for team in range(self.n_teams):
            while len(self.team_goals[team]) < self.n_goals_per_team:
                pos = (self.np_random.integers(self.grid_size), self.np_random.integers(self.grid_size))
                if pos not in taken:
                    taken.add(pos)
                    self.team_goals[team].append(pos)
        
        # Initialize previous positions for tracking
        self.prev_positions = [list(pos) for pos in self.agent_positions]
        
        obs = self._get_obs()
        return obs, {}
    
    def step(self, actions):
        """Execute actions for all agents and return observations, rewards, done flag"""
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        self.steps += 1
        
        # Track occupied positions
        occupied = {tuple(pos) for pos in self.agent_positions}
        new_positions = [list(pos) for pos in self.agent_positions]
        
        # Store distances before movement for reward shaping
        old_distances = []
        for i, pos in enumerate(self.agent_positions):
            team = i // self.n_agents_per_team
            distances = [abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]) for goal in self.team_goals[team]]
            old_distances.append(min(distances) if distances else 0)
        
        # Define movement directions and penalties
        move_dict = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        wall_penalty = -0.1
        collision_penalty = -0.2
        time_penalty = -0.01
        
        # Process agent movements
        for i, act in enumerate(actions):
            x, y = self.agent_positions[i]
            dx, dy = move_dict.get(act, (0, 0))
            nx, ny = x + dx, y + dy
            
            # Check wall collision
            if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
                rewards[i] += wall_penalty
                nx, ny = x, y
            # Check agent collision
            elif (nx, ny) not in occupied:
                occupied.remove((x, y))
                occupied.add((nx, ny))
                new_positions[i] = [nx, ny]
            else:
                rewards[i] += collision_penalty
                nx, ny = x, y
        
        # Update positions
        self.agent_positions = new_positions
        
        # Apply distance-based reward shaping
        shaping_factor = 0.5
        for i, pos in enumerate(self.agent_positions):
            team = i // self.n_agents_per_team
            distances = [abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]) for goal in self.team_goals[team]]
            new_dist = min(distances) if distances else 0
            
            # Only apply shaping if not adjacent to any goal
            near_goal = any(abs(pos[0] - g[0]) + abs(pos[1] - g[1]) <= 1 for g in self.team_goals[team])
            
            if not near_goal and old_distances[i] > 0 and new_dist >= 0:
                reward_shaping = (old_distances[i] - new_dist) * shaping_factor
                rewards[i] += reward_shaping
            
            # Time penalty for each step
            rewards[i] += time_penalty
        
        # Save positions for next step
        self.prev_positions = [list(pos) for pos in self.agent_positions]
        
        # Check goal collection
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
            
            # Remove collected goals
            for goal in goals_to_remove:
                self.team_goals[team].remove(goal)
        
        # Check if episode is done
        done = (self.steps >= self.max_steps) or all(len(goals) == 0 for goals in self.team_goals.values())
        obs = self._get_obs()
        
        return obs, rewards.tolist(), done, {}
    
    def _get_obs(self):
        """Generate observations for all agents"""
        obs = []
        
        for i, pos in enumerate(self.agent_positions):
            agent_obs = []
            
            # Agent's own position
            agent_obs.extend(pos)
            
            # Agent's team ID
            team_id = i // self.n_agents_per_team
            agent_obs.append(team_id)
            
            # All goals sorted by team and position for consistency
            goals = []
            for team in range(self.n_teams):
                sorted_goals = sorted(self.team_goals[team], key=lambda g: (g[0], g[1]))
                for goal_pos in sorted_goals:
                    goals.extend(goal_pos)
                # Pad with -1 for missing goals
                for _ in range(self.n_goals_per_team - len(sorted_goals)):
                    goals.extend([-1, -1])
            
            agent_obs.extend(goals)
            obs.append(np.array(agent_obs, dtype=np.int32))
        
        return obs