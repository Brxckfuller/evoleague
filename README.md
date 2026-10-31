# EvoLeague

Distributed self-play reinforcement learning with PyTorch.

EvoLeague is a machine learning engineering project exploring how populations of neural-network agents learn competitive strategies through self-play.

The project combines reinforcement learning, opponent leagues, multiprocessing and genuine distributed neural-network optimisation using PyTorch DistributedDataParallel (DDP).

## Research Question

Does training against a diverse population of historical opponents produce reinforcement learning policies that generalise better to unseen opponents than training exclusively against the most recent opponent?

## Planned Architecture

- Custom two-player territory-control environment
- PyTorch actor–critic neural network
- Proximal Policy Optimisation (PPO)
- Latest-opponent and historical-opponent self-play
- Multiple gameplay rollout workers
- Multiple learner processes using PyTorch DDP
- Controlled experiments measuring generalisation and training efficiency

## Development Progress

### Milestone 1 — Competitive Environment

Initial implementation completed:

- 7 × 7 territory-control arena
- Two players with alternating turns
- Five discrete actions
- Territory capture and scoring
- Match termination and draw detection
- Ten passing automated unit tests

The environment is an initial prototype. Reward attribution, collision rules and neural-network observation encoding are still being developed.

### Upcoming Milestones

1. Finalise the reinforcement learning environment.
2. Implement single-process PPO.
3. Introduce self-play and historical opponents.
4. Implement distributed PyTorch training with DDP.
5. Compare opponent-selection strategies.
6. Evaluate, visualise and document experimental results.

## Requirements

Python 3.12.

The initial environment uses only the Python standard library. PyTorch and additional dependencies will be introduced in later milestones.

## Run Tests

From the project root:

```bash
python -m unittest discover -s tests -v
```

## Project Status

Early development — Milestone 1.

The eventual goal is a reproducible distributed reinforcement learning system that runs locally on Apple Silicon hardware without requiring large datasets or cloud infrastructure.