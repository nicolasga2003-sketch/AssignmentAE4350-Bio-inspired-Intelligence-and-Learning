# Autonomous Lunar Lander Control (TD3)
**AE4350: Bio-inspired Intelligence and Learning for Aerospace Applications — TU Delft**

This repository contains a **Twin Delayed DDPG (TD3)** agent designed to autonomously land a spacecraft in the `LunarLanderContinuous-v3` environment. Transitioning from a standard DDPG baseline, this implementation solves common continuous control challenges like value overestimation and high-frequency actuator chattering.

### Key Features
* **Twin Critics & Policy Smoothing:** Stabilizes learning in continuous action spaces.
* **Dynamic Exploration Decay:** Transitions the agent from random system identification to precision terminal guidance.
* **Actuator Deadband Filter:** Neutralizes residual thrust upon touchdown to meet strict kinematic rest conditions and secure the maximum landing score.

### Repository Structure
* `requirements.txt`: Python dependencies.
* `networks.py`: Actor and Twin Critic neural network architectures.
* `agent.py`: Core TD3 algorithm and Experience Replay Buffer.
* `train.py`: Multi-seed training loop and learning curve generation.
* `evaluate.py`: Deterministic flight evaluation with deadband filter and video rendering.
