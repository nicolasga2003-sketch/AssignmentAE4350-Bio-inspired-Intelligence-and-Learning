import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from agent import TD3Agent, ReplayBuffer

def train_td3_seed(seed_value, max_episodes=800):
    env = gym.make("LunarLanderContinuous-v3")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Set seeds for reproducibility
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    agent = TD3Agent(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()

    max_timesteps = 500
    batch_size = 128
    exploration_noise = 0.20
    exploration_noise_min = 0.01
    exploration_decay = 0.995
    warmup_timesteps = 10000

    reward_history = []
    total_timesteps = 0

    for episode in range(max_episodes):
        state, _ = env.reset(seed=seed_value + episode)
        episode_reward = 0

        for t in range(max_timesteps):
            total_timesteps += 1

            if total_timesteps < warmup_timesteps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
                noise = np.random.normal(0, exploration_noise, size=action_dim)
                action = (action + noise).clip(-max_action, max_action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state
            episode_reward += reward

            if total_timesteps > warmup_timesteps:
                agent.train(replay_buffer, batch_size)

            if done:
                break

        if total_timesteps > warmup_timesteps:
            exploration_noise = max(exploration_noise_min, exploration_noise * exploration_decay)

        reward_history.append(episode_reward)

    env.close()
    
    # Save the model from the last seed to test it later
    torch.save(agent.actor.state_dict(), "td3_actor_lunarlander.pth")
    return reward_history

def smooth(data, window=10):
    """Applies a moving average to smooth the learning curve."""
    return np.convolve(data, np.ones(window)/window, mode='valid')

if __name__ == "__main__":
    seeds = [42, 100, 2024]
    all_rewards = []

    print("Initiating statistical evaluation... (This will take a few minutes)")
    for seed in seeds:
        print(f"--- Training seed {seed} ---")
        rewards = train_td3_seed(seed)
        all_rewards.append(rewards)

    # Compute mean and standard deviation
    rewards_np = np.array(all_rewards)
    rewards_mean = np.mean(rewards_np, axis=0)
    rewards_std = np.std(rewards_np, axis=0)

    smoothed_mean = smooth(rewards_mean)
    smoothed_std = smooth(rewards_std)
    episodes = np.arange(len(smoothed_mean))

    # Generate scientific plot
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, smoothed_mean, label='Mean Reward (TD3)', color='blue')
    plt.fill_between(episodes, 
                     smoothed_mean - smoothed_std, 
                     smoothed_mean + smoothed_std, 
                     color='blue', alpha=0.2, label='Standard Deviation')

    plt.axhline(y=200, color='r', linestyle='--', label='Successful Landing (+200)')
    plt.title("TD3 Performance on LunarLander (Multiple Seeds)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("learning_curve.png", dpi=300, bbox_inches='tight')
    print("Plot saved as 'learning_curve.png'.")
