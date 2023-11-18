import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from Agent.agent import Agent

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, 96)
        self.fc4 = nn.Linear(96, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DDQNAgent(Agent):
    def __init__(self, state_size, action_size, episodes, episode_length, train_interval=100, 
                 Update_target_frequency=100, memory_size=2000, gamma=0.95, 
                 learning_rate=0.001, batch_size=64, epsilon_min=0.01, train_test='train', symbol=""):
        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = memory_size
        self.memory = [None] * memory_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decrement = (self.epsilon - epsilon_min) * train_interval / (episodes * episode_length)
        self.learning_rate = learning_rate
        self.train_interval = train_interval
        self.Update_target_frequency = Update_target_frequency
        self.batch_size = batch_size
        self.brain = DQNNetwork(state_size, action_size)
        self.brain_ = DQNNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.brain.parameters(), lr=learning_rate)
        self.i = 0
        self.train_test = train_test
        self.symbol = symbol

    def save_model(self, path='./Saved Models/'):
        torch.save(self.brain.state_dict(), path + self.symbol + '.pt')

    def load_model(self, path='./Saved Models/'):
        self.brain.load_state_dict(torch.load(path + self.symbol + '.pt'))
        self.brain.eval()

    def _build_brain(self):
        return DQNNetwork(self.state_size, self.action_size)

    def act(self, state, test=False):
        action = np.zeros(self.action_size)
        if np.random.rand() <= self.epsilon and self.train_test == 'train' and not test:
            action[random.randrange(self.action_size)] = 1
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                act_values = self.brain(state).numpy()
            action[np.argmax(act_values[0])] = 1
        if test:
            return action, act_values
        else:
            return action

    def updateTargetModel(self):
        self.brain_.load_state_dict(self.brain.state_dict())

    def observe(self, state, action, reward, next_state, done, warming_up=False):
        self.i = (self.i + 1) % self.memory_size
        self.memory[self.i] = (state, action, reward, next_state, done)
        if (not warming_up) and (self.i % self.train_interval) == 0:
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decrement
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self._get_batches()
            state_batch = torch.from_numpy(state_batch).float()
            next_state_batch = torch.from_numpy(next_state_batch).float()
            reward_batch = torch.from_numpy(reward_batch).float()
            done_batch = torch.from_numpy(done_batch).float()

            self.optimizer.zero_grad()
            current_q_values = self.brain(state_batch)
            next_q_values = self.brain_(next_state_batch).detach()

            # get the q value for the action that was taken
            action_indices = torch.LongTensor(action_batch).unsqueeze(1)
            current_q_values = current_q_values.gather(1, action_indices).squeeze(1)

            # calculate the target q value
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - done_batch)

            # ensure no gradients are computed for target_q_values
            loss = nn.MSELoss()(current_q_values, target_q_values)
            loss.backward()
            self.optimizer.step()

            if self.i % self.Update_target_frequency == 0:
                self.updateTargetModel()

            return loss.item()

    def _get_batches(self):
        valid_memory = [m for m in self.memory if m is not None]
        batch = random.sample(valid_memory, min(len(valid_memory), self.batch_size))

        # put each experience in its own tuple
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        # turn each tuple of experiences into a batch
        state_batch = np.array(state_batch)
        action_batch = np.array([np.argmax(a) for a in action_batch])
        reward_batch = np.array(reward_batch)
        next_state_batch = np.array(next_state_batch)
        done_batch = np.array(done_batch, dtype=np.uint8)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

