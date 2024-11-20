import os
import json
from obstacles_board_generator import generate_obstacle_boards

def setup_environment():
    # Create directories
    os.makedirs('models/v17.1', exist_ok=True)
    os.makedirs('model_config', exist_ok=True)
    os.makedirs('images', exist_ok=True)
    
    # Create config file
    config = {
        "board_size": 10,
        "frames": 2,
        "max_time_limit": 998,
        "supervised": 0,
        "n_actions": 4,
        "obstacles": 1,
        "buffer_size": 100000,
        "gamma": 0.99,
        "learning_rate": 0.0001,
        "epsilon_decay": 0.997,
        "epsilon_min": 0.05,
        "use_target_net": True,
        "batch_size": 64,
        "target_update_freq": 200,
        "min_buffer_size": 5000
    }
    
    with open('model_config/v17.1.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Generate obstacle boards
    generate_obstacle_boards()
    
    print("Setup complete!")
    print("Created directories: models/v17.1, model_config, images")
    print("Created config file: model_config/v17.1.json")
    print("Generated obstacle boards")

if __name__ == "__main__":
    setup_environment()