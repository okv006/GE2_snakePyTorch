import numpy as np
from game_environment import Snake
from agent import DeepQLearningAgent
import json
import os
import matplotlib.pyplot as plt
from analyze_and_visualize import (record_best_episode, analyze_performance, save_with_metadata, create_game_visualization)


#loads config
print("Loading config...")
VERSION = 'v17.1'
#!!NB Local Path, please change before trying to run...
CONFIG_PATH = "c:/Users/O/Documents/Skole/03 Machine Learning/GE2/GE2_snake_PyTorch/model_config/v17.1.json" #change this, please

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

#setting defaults for missing config values
defaults = {
    'gamma': 0.99,
    'learning_rate': 0.0004,
    'epsilon_decay': 0.996,
    'epsilon_min': 0.01,
    'use_target_net': True
}

#updates config with defaults for missing values
for key, value in defaults.items():
    if key not in config:
        config[key] = value
        print(f"Using default value for {key}: {value}")

print("Config loaded with defaults")

#initializing environment 
env = Snake(
    board_size=config['board_size'],
    frames=config['frames'],
    max_time_limit=config['max_time_limit'],
    obstacles=bool(config['obstacles']),
    version=VERSION
)
state = env.reset()

#initializing agent
agent = DeepQLearningAgent(
    board_size=config['board_size'],
    frames=config['frames'],
    buffer_size=config['buffer_size'],
    gamma=config['gamma'],
    n_actions=env.get_num_actions(),
    use_target_net=config['use_target_net'],
    version=VERSION
)

def play_game(env, agent, training=True):
    """Playing one episode of the game"""
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    food_eaten = 0
    
    while not done:
        legal_moves = env.get_legal_moves()
        action = agent.move(state, legal_moves, env.get_values())
        next_state, reward, done, info, next_legal_moves = env.step(action)
        
        if training:
            #stores transition in replay buffer
            agent.add_to_buffer(state, action, reward, next_state, done, next_legal_moves)
            
            #trains more frequently early in the episode
            should_train = (
                len(agent.buffer) >= agent.min_buffer_size and
                (steps < 100 and steps % 2 == 0) or  #trains every 2 steps early on
                (steps >= 100 and steps % 4 == 0)    #trains every 4 steps later
            )
            
            if should_train:
                loss = agent.train_agent(batch_size=64)  #trains with batch size of 64
                if steps % 100 == 0:
                    print(f"Step {steps}, Loss: {loss:.4f}, Epsilon: {agent.epsilon:.4f}")
            
            #updates target network more frequently
            if steps % 200 == 0 and steps > 0:
                agent.update_target_net()
                print("Target network updated")
        
        #prints rewards or reason for term
        if reward > 0:
            food_eaten += 1
            print(f"Step {steps}: Food eaten. Total food: {food_eaten}")
        elif reward < 0:
            print(f"Step {steps}: Collision. Reason: {info['termination_reason']}")
        
        state = next_state
        total_reward += reward
        steps += 1
    
    return total_reward, steps, food_eaten, info['termination_reason']

#updates episode count and add better progress tracking
episodes = 100 
save_interval = 500
eval_interval = 100

training_rewards = []
training_steps = []
training_foods = []
best_eval_reward = float('-inf')
eval_frequency = 100  #evaluates every 100 episodes

print("\nStarting training...")
print(f"Using Version: {VERSION}")
print(f"Obstacles Enabled: {bool(config['obstacles'])}")
print(f"Total Episodes: {episodes}")
print(f"Initial Epsilon: {agent.epsilon}")

def plot_training_progress(rewards, steps, foods, window=100):
    plt.figure(figsize=(15, 5))
    
    #plots rewards
    plt.subplot(131)
    plt.plot(rewards)
    plt.plot(np.convolve(rewards, np.ones(window)/window, mode='valid'), 'r')
    plt.title('Rewards')
    plt.xlabel('Episode')
    
    # Plot steps
    plt.subplot(132)
    plt.plot(steps)
    plt.plot(np.convolve(steps, np.ones(window)/window, mode='valid'), 'r')
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    
    # Plot food
    plt.subplot(133)
    plt.plot(foods)
    plt.plot(np.convolve(foods, np.ones(window)/window, mode='valid'), 'r')
    plt.title('Food Collected')
    plt.xlabel('Episode')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()

def print_game_stats(episode, reward, steps, food, epsilon, buffer_size, total_episodes):
    """Print detailed game statistics"""
    print(f"\nEpisode {episode + 1}/{total_episodes}")
    print("-" * 40)
    print(f"Reward: {reward}")
    print(f"Steps: {steps}")
    print(f"Food Eaten: {food}")
    print(f"Epsilon: {epsilon:.4f}")
    print(f"Buffer Size: {buffer_size}")
    print("-" * 40)

def print_training_stats(rewards, steps, foods, last_n=100):
    """Print training statistics for the last n episodes"""
    if len(rewards) < last_n:
        return
        
    print(f"\nLast {last_n} Episodes Statistics:")
    print("-" * 40)
    print(f"Average Reward: {np.mean(rewards[-last_n:]):.2f}")
    print(f"Average Steps: {np.mean(steps[-last_n:]):.2f}")
    print(f"Average Food: {np.mean(foods[-last_n:]):.2f}")
    print(f"Max Food in Episode: {np.max(foods[-last_n:])}")
    print("-" * 40)

for episode in range(episodes):
    #trainings episode
    reward, steps, food, term_reason = play_game(env, agent, training=True)
    training_rewards.append(reward)
    training_steps.append(steps)
    training_foods.append(food)
    
    #prints episode stats
    print_game_stats(episode, reward, steps, food, agent.epsilon, len(agent.buffer), episodes)
    
    #plots progress every 100 episodes
    if (episode + 1) % 100 == 0:
        plot_training_progress(training_rewards, training_steps, training_foods)
    
    #evaluation phase
    if (episode + 1) % eval_frequency == 0:
        eval_rewards = []
        for _ in range(5):  #runs 5 evaluation episodes
            eval_reward, eval_steps, eval_food, _ = play_game(env, agent, training=False)
            eval_rewards.append(eval_reward)
        
        avg_eval_reward = np.mean(eval_rewards)
        print(f"\nEvaluation after {episode + 1} episodes:")
        print(f"Average Eval Reward: {avg_eval_reward:.2f}")
        
        #prints training stats
        print_training_stats(training_rewards, training_steps, training_foods)
        
        #saves if better
        if avg_eval_reward > best_eval_reward:
            best_eval_reward = avg_eval_reward
            agent.save_model(file_path='models', iteration=episode)
            print(f"New best model saved! Reward: {best_eval_reward:.2f}")

#after training is complete
print("\nTraining completed!")

#create analysis directory if it doesn't exist
os.makedirs('analysis', exist_ok=True)

#analyzes performance
stats = {
    'rewards': training_rewards,
    'steps': training_steps,
    'foods': training_foods
}
analyze_performance(training_rewards, training_steps, training_foods)

#records best episode
frames = record_best_episode(env, agent, 'analysis/best_episode.npy')

#saves best model with metadata
save_with_metadata(agent, stats, 'models/best_model.pt')

#creates visualization of best episode
create_game_visualization(frames, 'analysis/game_visualization.gif')

#final plot
plot_training_progress(training_rewards, training_steps, training_foods)

#save final training statistics
np.savez('training_stats.npz', 
         rewards=np.array(training_rewards),
         steps=np.array(training_steps),
         foods=np.array(training_foods))