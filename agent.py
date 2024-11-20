import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
import json
import os

class DuelingDQN(nn.Module):
    def __init__(self, input_channels, n_actions):
        super(DuelingDQN, self).__init__()
        #implements dueling dqn architecture which separates state-value and advantage estimation
        #feature extraction uses convolutional layers to process the game state
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  #normalizes activations to prevent internal covariate shift
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten()  #flattens the conv output for the fully connected layers
        )
        
        #calculates the flattened dimension for the fully connected layers
        self.fc_input_dim = self._get_conv_output_dim(input_channels)
        
        #value stream estimates the state value V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  #outputs a single value representing state value
        )
        
        #advantage stream estimates the advantage of each action A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)  #outputs advantage values for each possible action
        )
        
    def _get_conv_output_dim(self, input_channels):
        """Calculates the output dimension of the conv layers"""
        #uses a dummy input to calculate the conv output dimension dynamically
        dummy_input = torch.zeros(1, input_channels, 10, 10)
        conv_output = self.conv_layers(dummy_input)
        return conv_output.shape[1]
        
    def forward(self, x):
        features = self.conv_layers(x)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        #combines value and advantages using the dueling dqn formula: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        #subtracting mean advantage helps with stability and identifiability
        qvalues = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvalues

#named tuple for storing experience in replay buffer
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done', 'legal_moves'))

class PrioritizedReplayBuffer:
    def __init__(self, capacity, board_size, n_frames, n_actions, alpha=0.6):
        #implements prioritized experience replay (per) for more efficient learning
        self.capacity = capacity
        self.alpha = 0.6  #controls how much prioritization is used (0 = uniform sampling)
        self.beta = 0.4   #importance sampling weight to correct bias introduced by prioritized sampling
        self.beta_increment = 0.001  #gradually increases beta to 1 for unbiased sampling
        self.epsilon = 1e-6  #small constant to prevent zero probabilities
        self.experiences = []
        self.priorities = []
        self.position = 0
        
    def add(self, state, action, reward, next_state, done, legal_moves):
        #adds new experience with maximum priority for exploration
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.experiences) < self.capacity:
            self.experiences.append(Experience(state, action, reward, next_state, done, legal_moves))
            self.priorities.append(max_priority)
        else:
            #overwrites old experiences when buffer is full (circular buffer)
            self.experiences[self.position] = Experience(state, action, reward, next_state, done, legal_moves)
            self.priorities[self.position] = max_priority
            
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        if len(self.experiences) < batch_size:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
            
        #calculates sampling probabilities based on priorities
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()
        
        #samples experiences based on their priorities
        indices = np.random.choice(len(self.experiences), batch_size, p=probs)
        
        #unpacks experiences into separate arrays for batch processing
        batch = [self.experiences[idx] for idx in indices]
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])
        legal_moves = np.array([e.legal_moves for e in batch])
        self.beta = min(1.0, self.beta + self.beta_increment)

        return states, actions, rewards, next_states, dones, legal_moves
        
    def __len__(self):
        return len(self.experiences)


class DeepQLearningAgent:
    def __init__(self, board_size=10, frames=2, buffer_size=100000,
                 gamma=0.99, n_actions=4, use_target_net=True,
                 version='v17.1'):
        #initializes the deep q-learning agent with dueling architecture and prioritized replay
        self.board_size = board_size
        self.n_frames = frames  #number of consecutive frames used as input
        self.buffer_size = buffer_size
        self.n_actions = n_actions
        self.gamma = gamma  #discount factor for future rewards
        self.use_target_net = use_target_net  #uses separate target network for stability
        self.version = version
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #creates main policy network and optional target network
        self.policy_net = DuelingDQN(frames, n_actions).to(self.device)
        if self.use_target_net:
            self.target_net = DuelingDQN(frames, n_actions).to(self.device)
            self.update_target_net()
        
        #initializes adam optimizer with l2 regularization to prevent overfitting
        self.learning_rate = 0.00025
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5  #l2 regularization coefficient
        )
        
        #creates prioritized replay buffer for experience storage
        self.buffer = PrioritizedReplayBuffer(buffer_size, board_size, frames, n_actions)
        
        #epsilon-greedy exploration parameters
        self.epsilon = 1.0  #initial exploration rate
        self.epsilon_min = 0.02  #minimum exploration rate
        self.epsilon_decay = 0.9995  #decay rate for exploration
        self.min_buffer_size = 10000  #minimum experiences before training starts

    def update_target_net(self):
        """Updates target network by copying policy network weights"""
        if self.use_target_net:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def move(self, state, legal_moves, values=None):
        """Selects an action based on current policy"""
        #implements epsilon-greedy action selection with legal move masking
        if random.random() < self.epsilon:
            #random exploration considering only legal moves
            if legal_moves.ndim == 3:
                legal_moves = legal_moves.squeeze(0)
            possible_moves = np.where(legal_moves[0] == 1)[0]
            if len(possible_moves) == 0:  #fallback if no legal moves
                return np.random.randint(self.n_actions)
            return np.random.choice(possible_moves)
        
        #greedy action selection based on q-values
        with torch.no_grad():
            state = self._prepare_input(state)
            q_values = self.policy_net(state)
            #ensures legal_moves has correct shape
            if legal_moves.ndim == 3:
                legal_moves = legal_moves.squeeze(0)
            legal_moves = torch.FloatTensor(legal_moves).to(self.device)
            
            q_values = q_values.cpu().numpy()
            legal_moves = legal_moves.cpu().numpy()
            
            #masks illegal moves with negative infinity to ensure they're never selected
            masked_q_values = np.where(legal_moves == 1, q_values, float('-inf'))
            return np.argmax(masked_q_values, axis=1)[0]
    
    def _prepare_input(self, state):
        """Converts state to PyTorch format with correct channel ordering"""
        #handles single state and batch state preprocessing
        if state.ndim == 3:
            state = np.expand_dims(state, axis=0)
        
        #converts from nhwc to nchw format required by pytorch
        state = np.transpose(state, (0, 3, 1, 2))
        
        #normalizes values and converts to torch tensor
        state = state.astype(np.float32) / 4.0
        state = torch.FloatTensor(state).to(self.device)
        return state

    def train_agent(self, batch_size=128):
        """Trains the agent using a batch of experiences"""
        #waits for minimum buffer size before starting training
        if len(self.buffer) < self.min_buffer_size:
            return 0.0
            
        #samples batch of experiences from prioritized replay buffer
        states, actions, rewards, next_states, dones, legal_moves = self.buffer.sample(batch_size)
        
        #converts all inputs to torch tensors with appropriate shapes
        states = self._prepare_input(states)
        next_states = self._prepare_input(next_states)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        if legal_moves.ndim == 3:
            legal_moves = legal_moves.squeeze(1)
        legal_moves = torch.FloatTensor(legal_moves).to(self.device)
        
        #computes current q-values for taken actions
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        #computes target q-values using target network or policy network
        with torch.no_grad():
            if self.use_target_net:
                next_q_values = self.target_net(next_states)
            else:
                next_q_values = self.policy_net(next_states)
            
            #creates mask for legal moves to prevent selecting illegal actions
            legal_moves_mask = (legal_moves == 1)
            
            #applies legal moves mask to q-values
            next_q_values_masked = next_q_values.clone()
            next_q_values_masked[~legal_moves_mask] = float('-inf')
            
            #computes target q-values using bellman equation
            max_next_q_values = next_q_values_masked.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
            
        #computes mse loss and performs optimization step
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        #clips gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        #decays epsilon for less exploration over time
        if len(self.buffer) >= self.min_buffer_size:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()

    def add_to_buffer(self, state, action, reward, next_state, done, legal_moves):
        """Adds experience to replay buffer"""
        self.buffer.add(state, action, reward, next_state, done, legal_moves)


    def save_model(self, file_path='', iteration=None):
        """Saves the current models to disk"""
        #saves model state with optional iteration number
        if iteration is None:
            iteration = 0
        
        #creates save directory if it doesn't exist
        os.makedirs(file_path, exist_ok=True)
        
        #saves networks and training state
        save_path = f"{file_path}/model_{iteration:04d}.pt"
        state_dict = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'buffer_size': len(self.buffer)
        }
        
        if self.use_target_net:
            state_dict['target_net_state_dict'] = self.target_net.state_dict()
        
        torch.save(state_dict, save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, file_path='', iteration=None):
        """Loads models from disk"""
        if iteration is None:
            iteration = 0
        
        load_path = f"{file_path}/model_{iteration:04d}.pt"
        try:
            checkpoint = torch.load(load_path)
            #loads policy network
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            #loads target network if it exists
            if self.use_target_net and 'target_net_state_dict' in checkpoint:
                self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            #loads optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #loads training state
            self.epsilon = checkpoint['epsilon']
            
            print(f"Model loaded from {load_path}")
            print(f"Epsilon: {self.epsilon:.4f}")
            print(f"Buffer size was: {checkpoint['buffer_size']}")
            
        except FileNotFoundError:
            print(f"No model found at {load_path}")