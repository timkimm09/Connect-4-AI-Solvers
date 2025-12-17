# Connect 4 AI Solver

A comparative study of four AI approaches to playing Connect 4: Minimax with Alpha-Beta Pruning, Monte Carlo Tree Search, Reinforcement Learning, and Deterministic Heuristic Search.

## Overview

This project implements and evaluates four different AI agents for Connect 4. Each agent uses fundamentally different approaches to decision-making, ranging from exhaustive search to statistical sampling to learned value functions. Finally, we test these agents against each other in a round-robin tournament to determine which approach performs the best.

## Results

| Matchup | Winner | Win Rate | Loser | Win Rate | Draws |
|---------|--------|----------|-------|----------|-------|
| MCTS vs Minimax | Minimax | 48% | MCTS | 44% | 8% |
| MCTS vs Deterministic | MCTS | 60% | Deterministic | 40% | 0% |
| MCTS vs RL | MCTS | 100% | RL | 0% | 0% |
| Minimax vs Deterministic | Minimax | 100% | Deterministic | 0% | 0% |
| Minimax vs RL | Minimax | 100% | RL | 0% | 0% |
| Deterministic vs RL | Deterministic | 100% | RL | 0% | 0% |

## Directory Structure
```
├── connect4.py  
├── minimax_alphabeta_pruning.py 
├── mcts_approach.py         
├── mcts_fine_tuning.py        
├── rl_agent.py                
├── deterministic_agent.py      
├── compare_all_agents.py       
├── compare_test_rl_agent.py  
├── test_deterministic_agent.py
├── test_rl_agent.py           
├── play_ai.py               
├── play_mcts.py                
├── play_rl_vis.py            
├── play_rl_text.py           
├── requirements.txt         
└── Q_table_*.pickle       
```

## How to Run

### Clone the Repository
```bash
git clone # Preference-based HTTPS or SSH
cd connect4-ai-solver
```

### Install Dependencies

Requires Python 3.8+.
```bash
pip install -r requirements.txt
```

### Play Against an Agent

Play against Minimax:
```bash
python play_ai.py
```

Play against MCTS:
```bash
python play_mcts.py
```

Play against RL:
```bash
python play_rl_vis.py
```

Play against Deterministic Search:
```bash
python test_deterministic_agent.py
```

### Run the Tournament

Compare all agents against each other:
```bash
python compare_all_agents.py
```

## Agents

### Minimax with Alpha-Beta Pruning
Exhaustive depth-limited search with pruning. Uses a heuristic evaluation function that scores open 3-in-a-rows, 2-in-a-rows, and center column control. Default search depth is 6.

### Monte Carlo Tree Search
Statistical sampling approach using the Upper Confidence Bound (UCB1) for node selection. Optimized with bitboard representation with center-first move ordering and immediate win & block detection. Uses C=0.5, 500 simulations per move.

### Reinforcement Learning (Q-Learning)
Tabular Q-learning trained over 500,000 episodes against a heuristic opponent. Struggles against rarer positions due to limited Q-table coverage of the relatively large state space.

### Deterministic Heuristic
Evaluation function with single-step lookahead. Scores positions based on potential winning lines, center control, and immediate threat detections. Simple yet effective against weaker opponents.

## Dependencies

- gymnasium
- numpy
- pygame
- torch
- stable-baselines3
- tqdm
- matplotlib

## Collaborators

- Krish Angra
- Timothy Kim
- Jay Kalathur
- Raza Hlaing
