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
├── team_goal_env.py    # Gymnasium environment implementation
├── iql_agents.py       # Q-learning agents and controller
├── main.py             # Training script and visualization
├── renderer.py         # Pygame rendering for visualization
└── pyproject.toml      # Poetry dependencies
```

## How It Works

### Environment
- Grid world with randomly placed agents and goals
- Each team has specific colored goals to collect
- Episode ends when all goals are collected or max steps reached

### Observations
Each agent observes:
- Its own position (x, y)
- Its team ID
- Positions of all goals (sorted for consistency)

### Actions
- 0: Move up
- 1: Move down
- 2: Move left
- 3: Move right

### Rewards
- +10: Collecting a team goal
- -0.1: Hitting a wall
- -0.2: Colliding with another agent
- -0.01: Time penalty per step
- +0.5 × (distance_reduction): Moving closer to nearest goal

### Learning Algorithm
- Independent Q-Learning with epsilon-greedy exploration
- Linear epsilon decay from 1.0 to 0.01
- Learning rate: 0.1, Discount factor: 0.95

## Customization

### Modify Rewards
Edit `team_goal_env.py`:
- Goal collection reward: line ~135
- Collision penalties: lines ~88-90
- Distance-based shaping: lines ~115-126

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