# Multi-Agent Reinforcement Learning with Independent Q-Learning

A Python implementation of Independent Q-Learning (IQL) agents in a team-based grid world environment. Two teams of agents compete to collect their respective goals while learning cooperative behavior.

## Overview

This project demonstrates multi-agent reinforcement learning where agents must:
- Navigate a grid world environment
- Collect team-specific goals (blue team collects blue goals, orange team collects orange goals)
- Avoid collisions with walls and other agents
- Learn efficient paths through distance-based reward shaping

![Training Visualization](https://via.placeholder.com/600x300?text=Training+Visualization)
*Agents are shown as circles with letters (team 0: a,b,... | team 1: A,B,...)*

## Features

- **Team-based gameplay**: 2 teams with configurable agents per team
- **Independent learning**: Each agent maintains its own Q-table
- **Reward shaping**: Distance-based rewards guide agents toward goals
- **Real-time visualization**: Watch agents learn during the final episode
- **Performance tracking**: Plots showing team rewards and action distributions over time

## Installation

### Prerequisites
- Python 3.10 or higher
- Poetry for dependency management

### Setup

1. Clone the repository:
```bash
git clone https://github.com/thiagothomas/custom-marl-env.git
cd custom-marl-env
```

2. Install dependencies:
```bash
poetry install
```

## Usage

Run the training script:
```bash
poetry run python main.py
```

The script will:
1. Train agents for 10,000 episodes (configurable)
2. Display a visualization of the final episode
3. Show plots of team performance and action distributions

## Configuration

Modify parameters in `main.py`:

```python
# Environment parameters
grid_size = 8           # Size of the grid world
n_agents_per_team = 2   # Number of agents per team
max_steps = 100         # Maximum steps per episode
episodes = 10_000       # Number of training episodes
```

## Project Structure

```
├── team_goal_env_simple.py    # Simplified environment with state discretization
├── team_goal_env.py           # Original full-state environment
├── team_goal_env_improved.py  # Alternative with relative observations
├── iql_agents.py              # Q-learning agents and controller
├── main.py                    # Training script and visualization
├── renderer.py                # Pygame rendering for visualization
├── analyze_behavior.py        # Tools for analyzing agent behavior
├── compare_environments.py    # Environment comparison utilities
├── test_behavior.py           # Testing learned behaviors
└── pyproject.toml             # Poetry dependencies
```

## How It Works

### Environment (Simplified Version)
- 8x8 grid world divided into 2x2 regions (4 regions total)
- Randomly placed agents and team-specific goals
- Curriculum learning: starts with 1 goal per team, increases to 3
- Episode ends when all goals are collected or max steps reached
- Prevents column-sweeping behavior through state space reduction

### Actions
- 0: Move up
- 1: Move down
- 2: Move left
- 3: Move right

### Learning Algorithm
- Independent Q-Learning with epsilon-greedy exploration
- Linear epsilon decay from 1.0 to 0.01
- Learning rate: 0.1, Discount factor: 0.95

## Observation and Reward Structure

### Observations (Simplified Environment)

Each agent receives a 4-element observation vector:
```
[region_x, region_y, goal_direction, distance_category]
```

- **region_x, region_y**: The agent's current region (0 or 1 for each axis)
  - The 8x8 grid is divided into 2x2 regions (4 regions total)
  - Example: position (3,5) → region (0,1)

- **goal_direction**: Direction to nearest team goal (0-7)
  - 0 = East, 1 = NE, 2 = North, 3 = NW, 4 = West, 5 = SW, 6 = South, 7 = SE

- **distance_category**: How far to nearest team goal
  - 0 = Close (≤2 steps)
  - 1 = Medium (3-5 steps)  
  - 2 = Far (>5 steps)

**Note**: Agents only observe their own team's goals, not the opponent's.

### Reward Structure

#### Individual Agent Rewards:

1. **Goal Collection**:
   - Collector: +50
   - Teammates: +25 (50% sharing bonus)

2. **Exploration**: +2 for visiting a new region

3. **Proximity Bonuses**:
   - +5 for being adjacent to a goal (distance = 1)
   - +2 for being near a goal (distance = 2)
   - -0.5 penalty for being far from all goals (distance > 6)

4. **Movement Penalties**:
   - Wall collision: -1.0
   - Agent collision: -1.0
   - Time penalty: -0.1 per step

5. **Completion Bonus**: +(max_steps - steps_taken) × 0.5 when all goals collected

#### Example Scenario:
```
Team 0 (Blue): Agents a, b
Team 1 (Orange): Agents A, B

If agent 'a' collects a blue goal:
- Agent a: +50 (collector)
- Agent b: +25 (teammate bonus)
- Agents A, B: 0 (different team)

If agent 'a' is adjacent to a blue goal:
- Agent a: +5 - 0.1 = +4.9 net reward
```

### Curriculum Learning
The environment gradually increases difficulty:
- Starts with 1 goal per team
- Every 100 episodes, evaluates performance
- Increases up to 3 goals per team based on learning progress

This design encourages:
- **Exploration**: Bonuses prevent repetitive behavior
- **Cooperation**: Shared rewards for team success
- **Efficiency**: Time penalties reward quick solutions
- **Smart Navigation**: Directional observations guide goal-seeking

## Customization

### Modify Rewards
Edit `team_goal_env_simple.py`:
- Goal collection reward: line ~167
- Exploration bonus: line ~139
- Movement penalties: lines ~109-111
- Proximity bonuses: lines ~150-155
- Curriculum learning rate: line ~20

### Adjust Learning
Edit `iql_agents.py`:
- Q-learning update: line ~42
- Exploration strategy: line ~50

### Change Visualization
Edit `renderer.py`:
- Team colors: lines ~5-10
- Agent appearance: lines ~50-65

## Results

After training, you'll see:
1. **Team Reward Plot**: Shows how each team's performance improves over episodes
2. **Action Distribution Plot**: Reveals how agent behavior changes during training

Typical results show:
- Initial random exploration
- Gradual improvement as agents learn goal locations
- Stabilization as agents master efficient paths

## Dependencies

- `gymnasium`: RL environment framework
- `numpy`: Numerical computations
- `pygame`: Visualization
- `matplotlib`: Plotting results
- `pandas`: Data smoothing

## License

MIT License - feel free to use this code for your own projects!

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Built with Gymnasium and inspired by multi-agent reinforcement learning research.