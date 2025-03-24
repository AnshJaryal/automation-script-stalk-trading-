import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gym
from trading_env import StockTradingEnv

# Hyperparameters
GAMMA = 0.95
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = 10000
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = 1.0
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        return torch.argmax(self.model(state_tensor)).item()

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)

        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(dim=1)[0]
            target_q_values = rewards + (GAMMA * next_q_values * ~dones)

        current_q_values = self.model(states).gather(1, actions).squeeze()
        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

# Train the AI agent
def train_agent(env, agent, episodes=100):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.train()

        agent.update_target_model()
        agent.decay_epsilon()

        print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

    torch.save(agent.model.state_dict(), "trained_model.pth")  # Save trained model

# Test the AI agent on new stock data
def test_agent(env, agent, episodes=10):
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)  # Exploit learned strategy
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Test Episode {episode+1}/{episodes}, Reward: {total_reward:.2f}")

    print(f"Average Reward: {np.mean(total_rewards):.2f}")

# Main execution
if __name__ == "__main__":
    train_env = StockTradingEnv("data/TSLA_processed.csv")
    agent = DQNAgent(state_dim=4, action_dim=3)

    train_agent(train_env, agent, episodes=100)

    # Load trained model for testing
    agent.model.load_state_dict(torch.load("trained_model.pth"))

    test_env = StockTradingEnv("data/TSLA_processed.csv")  # Test on new stock data
    test_agent(test_env, agent, episodes=10)
