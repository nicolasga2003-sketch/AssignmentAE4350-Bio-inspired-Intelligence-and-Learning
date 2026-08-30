import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import numpy as np
from agent import TD3Agent

def evaluate_agent():
    print("Preparing environment for trained flight evaluation...")
    
    # Setup the environment to record the flight
    env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
    env = RecordVideo(env, video_folder="./video_test", name_prefix="trained_flight", disable_logger=True)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize agent and load pre-trained weights
    agent = TD3Agent(state_dim, action_dim, max_action)
    
    try:
        agent.actor.load_state_dict(torch.load("td3_actor_lunarlander.pth", weights_only=True))
        print("Trained model weights loaded successfully.")
    except FileNotFoundError:
        print("Error: Model weights not found. Please run train.py first.")
        return

    # Use a different seed for evaluation testing
    state, _ = env.reset(seed=100) 
    terminated = False
    truncated = False
    episode_reward = 0

    print("Initiating deterministic flight with Deadband Filter...")

    # Deterministic control loop
    while not (terminated or truncated):
        # The Actor computes the action directly based on the kinematic state
        action = agent.select_action(state)

        # DEADBAND FILTER IMPLEMENTATION
        # If the engine signal is below 10% (physical tolerance), shut down the engine
        action[abs(action) < 0.1] = 0.0

        # Apply commands to the physics model
        state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward

    print(f"Evaluation complete. Total Reward: {episode_reward:.2f}")
    env.close()

if __name__ == "__main__":
    evaluate_agent()
