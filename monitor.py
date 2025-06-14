from collections import deque
import numpy as np
import sys
import math

def interact(env, agent, num_episodes=20000, window=100):
    """Monitor agent performance and learning progress."""
    avg_rewards = deque(maxlen=num_episodes)
    best_avg_reward = -math.inf
    samp_rewards = deque(maxlen=window)

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        if isinstance(state, tuple):  # compatibility with newer Gym
            state = state[0]

        samp_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            step_output = env.step(action)

            # Handle older and newer Gym versions
            if len(step_output) == 4:
                next_state, reward, done, _ = step_output
            else:
                next_state, reward, terminated, truncated, _ = step_output
                done = terminated or truncated

            agent.step(state, action, reward, next_state, done)
            samp_reward += reward
            state = next_state

        samp_rewards.append(samp_reward)

        if i_episode >= window:
            avg_reward = np.mean(samp_rewards)
            avg_rewards.append(avg_reward)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward

        print(f"\rEpisode {i_episode}/{num_episodes} || Best average reward {best_avg_reward:.2f}", end="")
        sys.stdout.flush()

        if best_avg_reward >= 9.7:
            print(f"\nEnvironment solved in {i_episode} episodes!", end="")
            break

    print("\n")
    return avg_rewards, best_avg_reward
