''' Build the Policy Network. '''
import torch
from torch import nn

class ReinforcePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, state):
        dist = self.layers(state)
        return dist


''' Get some math for training. '''
from functools import partial
import numpy as np
import gymnasium as gym
from torch.distributions import Categorical

def discounted_reward(gamma: float, rewards: list[float], t: int) -> float:
    g = 0.
    for ti, r in reversed(list(enumerate(rewards))):
        if ti < t: break
        g = gamma * g + r
    return g

def run_trajectory(policy: nn.Module, env: gym.Env, *, gamma: float):
    episode_over = False
    total_reward = 0
    observation, _ = env.reset()
    rewards = []
    logps = []
    t = 0
    while not episode_over:
        observation /= np.array([4.8, 1, 0.418, 1])
        dist = Categorical(logits=policy(torch.as_tensor(observation)))
        action = dist.sample()
        observation, reward, terminated, truncated, _ = env.step(action.item())
        episode_over = terminated or truncated
        total_reward += float(reward)
        logp = dist.log_prob(action)
        logps.append(logp)
        rewards.append(float(reward))
        t += 1
    G = partial(discounted_reward, gamma, rewards)
    gs = torch.tensor([G(t) for t in range(t)])
    gs = torch.nn.functional.layer_norm(gs, (t,))
    logps = torch.hstack(logps)
    gammas = torch.tensor([gamma ** i for i in range(t)])
    l = (gs * gammas).dot(-logps)
    return l, total_reward


def train(policy: nn.Module, env: gym.Env, *, lr = 0.002, epochs = 130, batches = 10, gamma = 0.99):
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    for e in range(epochs):
        optimizer.zero_grad()
        L = torch.tensor(0.)
        tot_reward = 0.
        for _ in range(batches):
            l, r = run_trajectory(policy, env, gamma=gamma)
            L += l
            tot_reward += r
        L /= batches
        L.backward()
        optimizer.step()
        print(f"epoch: {e}; tot-reward: {tot_reward / batches}; loss: {L}")
    return policy

policy = ReinforcePolicy()

env = gym.make('CartPole-v1', render_mode=None)
train(policy, env)
env.close

env = gym.make('CartPole-v1', render_mode='human')
policy.eval()
print(run_trajectory(policy, env, gamma=0.99))
env.close
