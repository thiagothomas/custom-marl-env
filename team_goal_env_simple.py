import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TeamGoalEnvSimple(gym.Env):
    """
    Simplified environment with discretized state space and exploration bonuses.
    
    Key improvements:
    - Discretized position space (2x2 regions instead of 8x8 positions)
    - Exploration bonus for visiting new regions
    - Curriculum learning: starts with 1 goal per team, increases over time
    - Stronger directional rewards
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, grid_size=8, n_agents_per_team=2, max_steps=300, 
                 n_goals_per_team=3, curriculum_rate=0.001):
        super().__init__()
        self.grid_size = grid_size
        self.n_agents_per_team = n_agents_per_team
        self.n_teams = 2
        self.n_agents = self.n_agents_per_team * self.n_teams
        self.max_steps = max_steps
        self.max_goals_per_team = n_goals_per_team
        self.curriculum_rate = curriculum_rate
        
        # Curriculum learning: start with 1 goal
        self.current_goals_per_team = 1
        self.episodes_completed = 0
        
        # State discretization: divide grid into regions
        self.region_size = 4  # 8x8 grid -> 2x2 regions
        self.n_regions = self.grid_size // self.region_size
        
        # Observation: [region_x, region_y, nearest_goal_direction, distance_category]
        # direction: 0-7 (8 directions), distance: 0-2 (close/medium/far)
        obs_size = 4
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([self.n_regions-1, self.n_regions-1, 7, 2], dtype=np.float32)
        )
        
        # Action space: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)
        
        # Track exploration
        self.visited_regions = {}
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        
        # Update curriculum
        self.episodes_completed += 1
        if self.episodes_completed % 100 == 0:  # Check every 100 episodes
            progress = min(1.0, self.episodes_completed * self.curriculum_rate)
            self.current_goals_per_team = max(1, int(1 + progress * (self.max_goals_per_team - 1)))
        
        # Initialize agent positions randomly without overlap
        self.agent_positions = []
        taken = set()
        while len(self.agent_positions) < self.n_agents:
            pos = (self.np_random.integers(self.grid_size), self.np_random.integers(self.grid_size))
            if pos not in taken:
                taken.add(pos)
                self.agent_positions.append(list(pos))
        
        # Initialize team goals with curriculum
        self.team_goals = {team: [] for team in range(self.n_teams)}
        for team in range(self.n_teams):
            while len(self.team_goals[team]) < self.current_goals_per_team:
                pos = (self.np_random.integers(self.grid_size), self.np_random.integers(self.grid_size))
                if pos not in taken:
                    taken.add(pos)
                    self.team_goals[team].append(pos)
        
        # Reset exploration tracking for this episode
        self.visited_regions = {i: set() for i in range(self.n_agents)}
        
        # Mark initial regions as visited
        for i, pos in enumerate(self.agent_positions):
            region = self._get_region(pos)
            self.visited_regions[i].add(region)
        
        obs = self._get_obs()
        return obs, {}
    
    def _get_region(self, pos):
        """Convert position to region coordinates"""
        rx = pos[0] // self.region_size
        ry = pos[1] // self.region_size
        return (rx, ry)
    
    def _get_direction_and_distance(self, from_pos, to_pos):
        """Get direction (0-7) and distance category to target"""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        # Calculate direction (8 directions)
        angle = np.arctan2(dy, dx)
        direction = int((angle + np.pi) / (np.pi / 4)) % 8
        
        # Calculate distance category
        dist = abs(dx) + abs(dy)
        if dist <= 2:
            dist_cat = 0  # Close
        elif dist <= 5:
            dist_cat = 1  # Medium
        else:
            dist_cat = 2  # Far
        
        return direction, dist_cat
    
    def step(self, actions):
        """Execute actions for all agents and return observations, rewards, done flag"""
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        self.steps += 1
        
        # Track occupied positions
        occupied = {tuple(pos) for pos in self.agent_positions}
        new_positions = [list(pos) for pos in self.agent_positions]
        
        # Define movement and rewards
        move_dict = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        wall_penalty = -1.0
        collision_penalty = -1.0
        time_penalty = -0.1
        exploration_bonus = 2.0
        
        # Process agent movements
        for i, act in enumerate(actions):
            x, y = self.agent_positions[i]
            dx, dy = move_dict.get(act, (0, 0))
            nx, ny = x + dx, y + dy
            
            # Check wall collision
            if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
                rewards[i] += wall_penalty
            # Check agent collision
            elif (nx, ny) in occupied and (nx, ny) != (x, y):
                rewards[i] += collision_penalty
            else:
                # Valid move
                if (nx, ny) != (x, y):
                    occupied.remove((x, y))
                    occupied.add((nx, ny))
                new_positions[i] = [nx, ny]
                
                # Check for exploration bonus
                new_region = self._get_region([nx, ny])
                if new_region not in self.visited_regions[i]:
                    rewards[i] += exploration_bonus
                    self.visited_regions[i].add(new_region)
        
        # Update positions
        self.agent_positions = new_positions
        
        # Apply directional rewards
        for i, pos in enumerate(self.agent_positions):
            team = i // self.n_agents_per_team
            
            if self.team_goals[team]:
                # Find nearest goal
                distances = [(goal, abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])) 
                           for goal in self.team_goals[team]]
                nearest_goal, min_dist = min(distances, key=lambda x: x[1])
                
                # Strong reward for being adjacent to goal
                if min_dist == 1:
                    rewards[i] += 5.0
                elif min_dist == 2:
                    rewards[i] += 2.0
                
                # Penalty for being far from all goals
                if min_dist > 6:
                    rewards[i] += -0.5
            
            # Time penalty
            rewards[i] += time_penalty
        
        # Check goal collection
        goal_reward = 50.0  # Large goal reward
        for team in range(self.n_teams):
            team_agents = range(team * self.n_agents_per_team, (team + 1) * self.n_agents_per_team)
            goals_to_remove = []
            
            for goal_pos in self.team_goals[team]:
                for agent in team_agents:
                    if tuple(self.agent_positions[agent]) == goal_pos:
                        # Individual reward for collector, shared bonus for team
                        rewards[agent] += goal_reward
                        for teammate in team_agents:
                            if teammate != agent:
                                rewards[teammate] += goal_reward * 0.5
                        goals_to_remove.append(goal_pos)
                        break
            
            # Remove collected goals
            for goal in goals_to_remove:
                self.team_goals[team].remove(goal)
        
        # Bonus for completing all goals quickly
        if all(len(goals) == 0 for goals in self.team_goals.values()):
            time_bonus = max(0, (self.max_steps - self.steps) * 0.5)
            for i in range(self.n_agents):
                rewards[i] += time_bonus
        
        # Check if episode is done
        done = (self.steps >= self.max_steps) or all(len(goals) == 0 for goals in self.team_goals.values())
        obs = self._get_obs()
        
        return obs, rewards.tolist(), done, {}
    
    def _get_obs(self):
        """Generate simplified observations for all agents"""
        obs = []
        
        for i, pos in enumerate(self.agent_positions):
            # Get agent's region
            region = self._get_region(pos)
            
            # Get agent's team
            team = i // self.n_agents_per_team
            
            # Find nearest goal direction and distance
            if self.team_goals[team]:
                distances = [(goal, abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])) 
                           for goal in self.team_goals[team]]
                nearest_goal, _ = min(distances, key=lambda x: x[1])
                direction, dist_cat = self._get_direction_and_distance(pos, nearest_goal)
            else:
                # No goals left
                direction, dist_cat = 0, 0
            
            obs.append(np.array([region[0], region[1], direction, dist_cat], dtype=np.float32))
        
        return obs
    
    @property
    def n_goals_per_team(self):
        """For compatibility with renderer"""
        return self.max_goals_per_team