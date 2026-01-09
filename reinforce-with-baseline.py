''' Build the Policy Network. '''
import torch
from torch import nn

class ReinforcePolicy(nn.Module):
    def __init__(self) -> None:
        ''' Simply share model backbone between that actor & the critic. '''
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, state):
        dist = self.layers(state)
        logits, value = dist[:2], dist[2]
        return logits, value


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
    vs = []
    t = 0
    while not episode_over:
        observation /= np.array([4.8, 1, 0.418, 1])
        logits, value = policy(torch.as_tensor(observation))
        dist = Categorical(logits=logits)
        action = dist.sample()
        observation, reward, terminated, truncated, _ = env.step(action.item())
        episode_over = terminated or truncated
        total_reward += float(reward)
        logp = dist.log_prob(action)
        logps.append(logp)
        rewards.append(float(reward))
        vs.append(value)
        t += 1
    G = partial(discounted_reward, gamma, rewards)
    gs = torch.tensor([G(t) for t in range(t)])
    vs = torch.hstack(vs)  # keep the grad
    # As = torch.nn.functional.layer_norm(gs - vs, (t,))  # this will destroy training
    logps = torch.hstack(logps)
    gammas = torch.tensor([gamma ** i for i in range(t)])
    loss = - (gammas * logps).dot(gs - vs) + 0.4 * nn.MSELoss()(gs, vs)  # policy reward & value estimator
    return loss, total_reward


def train(policy: nn.Module, env: gym.Env, *, lr = 0.003, epochs = 130, batches = 10, gamma = 0.99):
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    for e in range(epochs):
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.)
        batch_reward = 0.
        for _ in range(batches):
            loss, reward = run_trajectory(policy, env, gamma=gamma)
            batch_loss += loss; batch_reward += reward
        batch_loss /= batches; batch_reward /= batches
        print(f"epoch: {e}; tot-reward: {batch_reward}; loss: {batch_loss}")
        batch_loss.backward()
        optimizer.step()
    return policy

policy = ReinforcePolicy()

env = gym.make('CartPole-v1', render_mode=None)
train(policy, env)
env.close

env = gym.make('CartPole-v1', render_mode='human')
policy.eval()
print(run_trajectory(policy, env, gamma=0.99))
env.close
