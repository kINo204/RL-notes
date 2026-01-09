import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

env = gym.make("FetchPickAndPlaceDense-v4", render_mode='human')
env.reset()

episode_over = False
total_reward = 0

while not episode_over:
    action = env.action_space.sample()  # Random action for now - real agents will be smarter!
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward  # type: ignore
    episode_over = terminated or truncated
    env.render()

env.close()