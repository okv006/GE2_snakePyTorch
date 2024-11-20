"""This module stores the game environment. Note that the snake is a part of
the environment itself in this implementation.
"""

import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pickle

class Position:
    """Class for defining any position on a 2D grid"""
    def __init__(self, row=0, col=0):
        self.row = row
        self.col = col

    def set_position(self, row=None, col=None):
        if row is not None:
            self.row = row
        if col is not None:
            self.col = col

class Snake:
    """Class for the snake game."""
    def __init__(self, board_size=10, frames=2, start_length=5, seed=42,
                 max_time_limit=298, obstacles=False, version=''):
        self._value = {'snake':1, 'board':0, 'food':3, 'head':2, 'border':4}
        self._actions = {-1:-1, 0:0, 1:1, 2:2, 3:3, 4:-1}
        self._n_actions = 4
        self._board_size = board_size
        self._n_frames = frames
        self._rewards = {
            'out': -1,        #keep death penalty
            'food': 2,        #increase food reward
            'time': 0.01,     #small reward for survival
            'no_food': -0.5,  #penalty for not eating
            'closer': 0.1,    #reward for getting closer to food
            'further': -0.1   #penalty for moving away from food
        }
        self._start_length = 2
        self._max_time_limit = max_time_limit
        self._obstacles = obstacles
        self._version = version
        self._get_static_board_template()

    def _get_static_board_template(self):
        """Creates the static board template."""
        if not self._obstacles:
            self._static_board_template = self._value['board'] * np.ones((self._board_size, self._board_size))
            self._static_board_template[:, 0] = self._value['border']
            self._static_board_template[:, self._board_size-1] = self._value['border']
            self._static_board_template[0, :] = self._value['border']
            self._static_board_template[self._board_size-1, :] = self._value['border']
        else:
            try:
                #try multiple path variants
                possible_paths = [
                    f'models/v17.1/obstacles_board',  #direct path
                    f'models/{self._version}/obstacles_board',  
                    'obstacles_board'  
                ]
                
                found = False
                for path in possible_paths:
                    try:
                        with open(path, 'rb') as f:
                            self._static_board_template = pickle.load(f)
                            found = True
                            print(f"Successfully loaded obstacles from: {path}")
                            break
                    except FileNotFoundError:
                        continue
                
                if not found:
                    raise FileNotFoundError("Could not find obstacles_board file in any expected location")
                    
                self._static_board_template = self._static_board_template[
                    np.random.choice(self._static_board_template.shape[0], 1), :, :]
                self._static_board_template = self._static_board_template.reshape((self._board_size, -1))
                self._static_board_template *= self._value['border']
                
            except Exception as e:
                print(f"Error loading obstacles: {str(e)}")
                print("Falling back to default border-only board...")
                self._static_board_template = self._value['board'] * np.ones((self._board_size, self._board_size))
                self._static_board_template[:, 0] = self._value['border']
                self._static_board_template[:, self._board_size-1] = self._value['border']
                self._static_board_template[0, :] = self._value['border']
                self._static_board_template[self._board_size-1, :] = self._value['border']

    def reset(self):
        """Resets the environment to the starting state."""
        self._get_static_board_template()
        board = self._static_board_template.copy()
        
        # Initialize snake
        self._snake = deque()
        self._snake_length = self._start_length
        self._count_food = 0
        
        # Place snake horizontally in middle
        for i in range(1, self._snake_length+1):
            board[self._board_size//2, i] = self._value['snake']
            self._snake.append(Position(self._board_size//2, i))
        
        # Set snake head
        self._snake_head = Position(self._board_size//2, i)
        board[self._snake_head.row, self._snake_head.col] = self._value['head']
        
        # Initialize board queue
        self._board = deque(maxlen=self._n_frames)
        for i in range(self._n_frames):
            self._board.append(board.copy())

        # Add food and initialize direction
        self._get_food()
        self._snake_direction = 0
        self._time = 0
        
        return self._queue_to_board()

    def _queue_to_board(self):
        """Convert queue of frames to 3D matrix."""
        board = np.dstack([x for x in self._board])
        return board.copy()

    def _get_food(self):
        """Find coordinates for new food placement."""
        ord_x = list(range(1, self._board_size-1))
        np.random.shuffle(ord_x)
        found = False
        
        for x in ord_x:
            food_y = [i for i in range(1, self._board_size-1) 
                     if self._board[0][x, i] == self._value['board']]
            if len(food_y) > 0:
                food_y = np.random.choice(food_y)
                self._food = Position(x, food_y)
                self._put_food()
                found = True
                break

    def _put_food(self):
        """Place food on the board."""
        self._board[0][self._food.row, self._food.col] = self._value['food']

    def get_num_actions(self):
        """Returns number of possible actions."""
        return self._n_actions

    def get_board_size(self):
        """Returns the board size."""
        return self._board_size

    def get_n_frames(self):
        """Returns number of frames."""
        return self._n_frames

    def get_values(self):
        """Returns the value dictionary."""
        return self._value

    def get_legal_moves(self):
        """Returns array of legal moves."""
        legal_moves = np.ones((1, self._n_actions), dtype=np.uint8)
        legal_moves[0, (self._snake_direction-2)%4] = 0
        return legal_moves.copy()

    def _action_map(self, action):
        """Maps action to internal representation."""
        return self._actions[action]

    def _get_new_direction(self, action, current_direction):
        """Calculate new direction based on action."""
        if self._action_map(action) == -1:
            return current_direction
        elif abs(self._action_map(action) - current_direction) == 2:
            return current_direction
        else:
            return self._action_map(action)

    def _get_new_head(self, action, current_direction):
        """Calculate new head position."""
        new_dir = self._get_new_direction(action, current_direction)
        if new_dir == 0:
            del_x, del_y = 1, 0
        elif new_dir == 1:
            del_x, del_y = 0, 1
        elif new_dir == 2:
            del_x, del_y = -1, 0
        else:
            del_x, del_y = 0, -1
            
        new_head = Position(
            self._snake_head.row - del_y,
            self._snake_head.col + del_x
        )
        return new_head

    def step(self, action):
        """Takes an action and returns new state, reward, done, and info."""
        reward, done = 0, 0

        if isinstance(action, np.ndarray):
            action = int(action[0])

        reward, done, can_eat_food, termination_reason = self._check_if_done(action)
        
        if done == 0:
            self._move_snake(action, can_eat_food)
            self._snake_direction = self._get_new_direction(action, self._snake_direction)
            if can_eat_food:
                self._get_food()

        self._time += 1
        info = {
            'time': self._time,
            'food': self._count_food,
            'termination_reason': termination_reason
        }
        
        next_legal_moves = self.get_legal_moves().copy()
        
        return self._queue_to_board(), reward, done, info, next_legal_moves

    def _check_if_done(self, action):
        """Check if game is over and calculate reward."""
        reward, done, can_eat_food, termination_reason = self._rewards['time'], 0, 0, ''
        new_head = self._get_new_head(action, self._snake_direction)
        
        # Check various termination conditions
        if self._board[0][new_head.row, new_head.col] == self._value['border']:
            done = 1
            reward = self._rewards['out']
            termination_reason = 'collision_wall'
        elif self._board[0][new_head.row, new_head.col] == self._value['snake']:
            snake_tail = self._snake[0]
            if not (new_head.row == snake_tail.row and new_head.col == snake_tail.col):
                done = 1
                reward = self._rewards['out']
                termination_reason = 'collision_self'
        elif self._board[0][new_head.row, new_head.col] == self._value['food']:
            reward += self._rewards['food']
            self._count_food += 1
            can_eat_food = 1
        
        if self._time >= self._max_time_limit and self._max_time_limit != -1:
            done = 1
            if self._snake_length == self._start_length and self._rewards['no_food'] != 0:
                termination_reason = 'time_up_no_food'
                reward += self._rewards['no_food']
            else:
                termination_reason = 'time_up'
                
        return reward, done, can_eat_food, termination_reason

    def _move_snake(self, action, can_eat_food):
        """Move the snake based on action."""
        new_head = self._get_new_head(action, self._snake_direction)
        new_board = self._board[0].copy()
        new_board[self._snake_head.row, self._snake_head.col] = self._value['snake']
        
        self._snake.append(new_head)
        self._snake_head = new_head

        if can_eat_food:
            self._snake_length += 1
        else:
            delete_pos = self._snake.popleft()
            new_board[delete_pos.row, delete_pos.col] = self._value['board']
            
        new_board[new_head.row, new_head.col] = self._value['head']
        self._board.appendleft(new_board.copy())