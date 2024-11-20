import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from datetime import datetime
import torch

def record_best_episode(env, agent, save_path):
    """Record the best episode for visualization"""
    state = env.reset()
    done = False
    total_reward = 0
    frames = []
    food_eaten = 0
    
    while not done:
        frames.append(state.copy())
        legal_moves = env.get_legal_moves()
        action = agent.move(state, legal_moves, env.get_values())
        next_state, reward, done, info, _ = env.step(action)
        
        state = next_state
        total_reward += reward
        if reward > 0:
            food_eaten += 1
            
    print(f"Recorded episode - Food: {food_eaten}, Reward: {total_reward}")
    
    # Save frames
    np.save(save_path, np.array(frames))
    return frames

def create_game_visualization(frames, save_path, fps=10):
    """Create video visualization from saved frames"""
    # Create directory if it doesn't exist
    if not os.path.isabs(save_path):
        project_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(project_dir, save_path)
            
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def update(frame_num):
        ax.clear()
        frame = frames[frame_num]
        
        # Custom colormap
        colors = {
            0: 'white',      # Empty space
            1: 'lightgreen', # Snake body
            2: 'darkgreen',  # Snake head
            3: 'red',        # Food
            4: 'gray'        # Walls/obstacles
        }
        
        cmap = plt.cm.colors.ListedColormap(list(colors.values()))
        
        im = ax.imshow(frame[:,:,0], cmap=cmap, vmin=0, vmax=4)
        ax.grid(True, which='major', color='black', linewidth=0.5, alpha=0.3)
        ax.set_title(f'Frame {frame_num}')
        return [im]
    
    anim = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=1000/fps, blit=True
    )
    
    # Save animation as GIF
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()
    
    print(f"Animation saved to: {save_path}")

def analyze_performance(training_rewards, training_steps, training_foods):
    """Analyze training performance"""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(project_dir, 'analysis', 'performance_analysis.png')
    
    plt.figure(figsize=(15, 10))

    
    # Plot moving averages
    window = 100
    plt.subplot(2, 2, 1)
    plt.plot(np.convolve(training_rewards, np.ones(window)/window, mode='valid'))
    plt.title('Average Reward over Time')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(2, 2, 2)
    plt.plot(np.convolve(training_steps, np.ones(window)/window, mode='valid'))
    plt.title('Average Steps over Time')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.subplot(2, 2, 3)
    plt.plot(np.convolve(training_foods, np.ones(window)/window, mode='valid'))
    plt.title('Average Food Collected over Time')
    plt.xlabel('Episode')
    plt.ylabel('Food Count')
    
    # Performance distribution
    plt.subplot(2, 2, 4)
    plt.hist(training_foods, bins=50)
    plt.title('Distribution of Food Collection')
    plt.xlabel('Food Count')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Performance analysis saved to: {save_path}")

def save_with_metadata(agent, stats, file_path='models/best_model.pt'):
    """Save model with performance metadata"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    state_dict = {
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict() if agent.use_target_net else None,
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'performance': {
            'avg_reward': np.mean(stats['rewards'][-100:]),
            'avg_steps': np.mean(stats['steps'][-100:]),
            'avg_food': np.mean(stats['foods'][-100:]),
            'max_food': np.max(stats['foods']),
            'training_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    torch.save(state_dict, file_path)
    print(f"Model saved with metadata to {file_path}")