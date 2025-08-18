import time
import h5py
import numpy as np
from typing import Dict, Any, List
import json
import os
from pathlib import Path

class ProprioceptiveDataLogger:
    """
    Logger for proprioceptive state data based on AgiBotDigitalWorld format.
    """
    
    def __init__(self, task_id: str = "001", episode_id: str = "auto", 
                 base_data_path: str = "/home/simt-wj/01_Active/smolvla_ws/data_atsume/data"):
        self.task_id = task_id
        self.base_data_path = Path(base_data_path)
        
        # Auto-increment episode_id if set to "auto"
        if episode_id == "auto":
            self.episode_id = self._get_next_episode_id()
        else:
            self.episode_id = episode_id
        
        # Create the full directory structure
        self.proprio_dir = self.base_data_path / "proprio_stats" / task_id / self.episode_id
        self.proprio_dir.mkdir(parents=True, exist_ok=True)
        
        # Set the filename with full path
        self.filename = self.proprio_dir / "proprio_stats.h5"
        
        # Create other directories in the structure
        self.task_info_dir = self.base_data_path / "task_info"
        self.observations_dir = self.base_data_path / "observations" / task_id / self.episode_id
        
        # Create all necessary directories
        self.task_info_dir.mkdir(parents=True, exist_ok=True)
        self.observations_dir.mkdir(parents=True, exist_ok=True)
        (self.observations_dir / "depth").mkdir(exist_ok=True)
        (self.observations_dir / "videos").mkdir(exist_ok=True)
        
        # Create task info file
        self.create_task_info()
        
        self.data_buffer = {
            'timestamp': [],
            'state': {
                'effector': {'position': []},
                'end': {'position': [], 'orientation': [], 'velocity': [], 'angular': []},
                'joint': {'position': [], 'velocity': [], 'effort': []},
                'robot': {'position': [], 'orientation': []}
            },
            'action': {
                'effector': {'position': [], 'index': []},
                'end': {'position': [], 'orientation': [], 'index': []},
                'joint': {'position': [], 'index': []},
                'robot': {'velocity': [], 'index': []}
            }
        }
        self.action_counter = 0
    
    def _get_next_episode_id(self) -> str:
        """
        Get the next available episode ID by checking existing episodes.
        Returns a zero-padded string like "001", "002", etc.
        """
        # Check both proprio_stats and observations directories for existing episodes
        proprio_task_dir = self.base_data_path / "proprio_stats" / self.task_id
        obs_task_dir = self.base_data_path / "observations" / self.task_id
        
        max_episode_num = 0
        
        # Check existing episodes in proprio_stats directory
        if proprio_task_dir.exists():
            for episode_dir in proprio_task_dir.iterdir():
                if episode_dir.is_dir():
                    try:
                        episode_num = int(episode_dir.name)
                        max_episode_num = max(max_episode_num, episode_num)
                    except ValueError:
                        continue
        
        # Check existing episodes in observations directory
        if obs_task_dir.exists():
            for episode_dir in obs_task_dir.iterdir():
                if episode_dir.is_dir():
                    try:
                        episode_num = int(episode_dir.name)
                        max_episode_num = max(max_episode_num, episode_num)
                    except ValueError:
                        continue
        
        # Check task info file for existing episodes
        task_info_file = self.base_data_path / "task_info" / f"task_{self.task_id}.json"
        if task_info_file.exists():
            try:
                with open(task_info_file, 'r') as f:
                    task_info = json.load(f)
                
                for episode in task_info.get("episodes", []):
                    try:
                        episode_num = int(episode["episode_id"])
                        max_episode_num = max(max_episode_num, episode_num)
                    except (ValueError, KeyError):
                        continue
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # Return next episode ID with zero padding
        next_episode_num = max_episode_num + 1
        return f"{next_episode_num:03d}"
    
    def create_task_info(self) -> None:
        """Create task info JSON file with basic information or add to existing one."""
        task_info_file = self.task_info_dir / f"task_{self.task_id}.json"
        
        # New episode info
        new_episode = {
            "episode_id": self.episode_id,
            "language_instruction": f"Demonstrative episode {self.episode_id} for task {self.task_id}",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": None,  # Will be updated when logging is complete
            "total_timesteps": None    # Will be updated when logging is complete
        }
        
        # Check if task info file already exists
        if task_info_file.exists():
            try:
                with open(task_info_file, 'r') as f:
                    task_info = json.load(f)
                
                # Check if this episode already exists
                existing_episode = False
                for episode in task_info.get("episodes", []):
                    if episode.get("episode_id") == self.episode_id:
                        existing_episode = True
                        break
                
                # Add new episode if it doesn't exist
                if not existing_episode:
                    task_info["episodes"].append(new_episode)
                    print(f"Added episode {self.episode_id} to existing task {self.task_id}")
                else:
                    print(f"Episode {self.episode_id} already exists in task {self.task_id}")
                    
            except (json.JSONDecodeError, KeyError):
                # If file is corrupted, create new task info
                task_info = self._create_new_task_info(new_episode)
                print(f"Recreated corrupted task info for task {self.task_id}")
        else:
            # Create new task info file
            task_info = self._create_new_task_info(new_episode)
            print(f"Created new task info for task {self.task_id}")
        
        # Save the updated task info
        with open(task_info_file, 'w') as f:
            json.dump(task_info, f, indent=2)
        
        print(f"Task info saved: {task_info_file}")
    
    def _create_new_task_info(self, episode_info: dict) -> dict:
        """Create a new task info structure."""
        return {
            "task_id": self.task_id,
            "episodes": [episode_info],
            "description": f"Proprioceptive data collection for task {self.task_id}",
            "robot_config": {
                "dof": 34,
                "arms": 2,
                "grippers": 2,
                "sensors": ["joint_position", "joint_velocity", "joint_effort", "end_effector_pose", "gripper_position"]
            }
        }
    
    def update_task_info(self, duration: float, timesteps: int) -> None:
        """Update task info with completion data."""
        task_info_file = self.task_info_dir / f"task_{self.task_id}.json"
        
        if task_info_file.exists():
            with open(task_info_file, 'r') as f:
                task_info = json.load(f)
            
            # Update the episode info
            for episode in task_info["episodes"]:
                if episode["episode_id"] == self.episode_id:
                    episode["duration_seconds"] = duration
                    episode["total_timesteps"] = timesteps
                    break
            
            with open(task_info_file, 'w') as f:
                json.dump(task_info, f, indent=2)
    
    def get_current_timestamp(self) -> float:
        """Get current timestamp in simulation time format."""
        return time.time()
    
    def get_gripper_state(self) -> np.ndarray:
        """
        Get gripper positions for left and right grippers.
        Returns: [N, 2] array with left [:, 0], right [:, 1] in mm
        """
        # Simulate gripper open range 0-100mm
        return np.random.uniform(0, 100, size=2)
    
    def get_end_effector_state(self) -> Dict[str, np.ndarray]:
        """
        Get end effector state for both arms.
        Returns: Dictionary with position, orientation, velocity, angular
        """
        # Position for left and right arms [2, 3]
        position = np.random.uniform([-0.5, -0.5, 0.0], [0.5, 0.5, 1.0], size=(2, 3))
        
        # Orientation quaternions for left and right arms [2, 4] (wxyz format)
        orientation = np.random.normal(0, 1, size=(2, 4))
        orientation[0] /= np.linalg.norm(orientation[0])
        orientation[1] /= np.linalg.norm(orientation[1])
        
        # Velocity for both arms [2, 3]
        velocity = np.random.uniform(-0.1, 0.1, size=(2, 3))
        
        # Angular velocity for both arms [2, 3]
        angular = np.random.uniform(-0.5, 0.5, size=(2, 3))
        
        return {
            'position': position,
            'orientation': orientation,
            'velocity': velocity,
            'angular': angular
        }
    
    def get_joint_state(self) -> Dict[str, np.ndarray]:
        """
        Get joint state for 34 DOF robot.
        Returns: Dictionary with position, velocity, effort
        """
        # 34 DOF: 2 head + 2 waist + 7*2 arms + 8*2 grippers
        position = np.random.uniform(-np.pi, np.pi, size=34)
        velocity = np.random.uniform(-1.0, 1.0, size=34)
        effort = np.random.uniform(-10.0, 10.0, size=34)
        
        return {
            'position': position,
            'velocity': velocity,
            'effort': effort
        }
    
    def get_robot_state(self) -> Dict[str, np.ndarray]:
        """
        Get robot pose in odometry coordinate system.
        Returns: Dictionary with position, orientation
        """
        # Position [3] where z is always 0
        position = np.array([
            np.random.uniform(-5, 5),  # x
            np.random.uniform(-5, 5),  # y
            0.0                        # z always 0
        ])
        
        # Orientation quaternion [4] (wxyz format)
        orientation = np.random.normal(0, 1, size=4)
        orientation /= np.linalg.norm(orientation)
        
        return {
            'position': position,
            'orientation': orientation
        }
    
    def generate_action_commands(self) -> Dict[str, Any]:
        """
        Generate action commands with appropriate indices.
        """
        actions = {}
        
        # Effector actions (gripper commands)
        if np.random.random() > 0.7:  # 30% chance to send gripper command
            actions['effector'] = {
                'position': np.random.uniform(0, 100, size=2),
                'index': [self.action_counter]
            }
        
        # End effector actions
        if np.random.random() > 0.8:  # 20% chance to send end effector command
            position = np.random.uniform([-0.5, -0.5, 0.0], [0.5, 0.5, 1.0], size=(2, 3))
            orientation = np.random.normal(0, 1, size=(2, 4))
            orientation[0] /= np.linalg.norm(orientation[0])
            orientation[1] /= np.linalg.norm(orientation[1])
            
            actions['end'] = {
                'position': position,
                'orientation': orientation,
                'index': [self.action_counter]
            }
        
        # Joint actions (only 14 DOF for joint position control)
        if np.random.random() > 0.6:  # 40% chance to send joint command
            actions['joint'] = {
                'position': np.random.uniform(-np.pi, np.pi, size=14),
                'index': [self.action_counter]
            }
        
        # Robot base actions
        if np.random.random() > 0.9:  # 10% chance to send robot base command
            actions['robot'] = {
                'velocity': np.array([
                    np.random.uniform(-0.5, 0.5),  # x velocity
                    np.random.uniform(-0.5, 0.5)   # yaw rate
                ]),
                'index': [self.action_counter]
            }
        
        if actions:  # Only increment if we actually sent an action
            self.action_counter += 1
        
        return actions
    
    def log_step(self) -> None:
        """Log one step of proprioceptive data."""
        # Get timestamp
        timestamp = self.get_current_timestamp()
        self.data_buffer['timestamp'].append(timestamp)
        
        # Get state data
        gripper_state = self.get_gripper_state()
        self.data_buffer['state']['effector']['position'].append(gripper_state)
        
        end_state = self.get_end_effector_state()
        self.data_buffer['state']['end']['position'].append(end_state['position'])
        self.data_buffer['state']['end']['orientation'].append(end_state['orientation'])
        self.data_buffer['state']['end']['velocity'].append(end_state['velocity'])
        self.data_buffer['state']['end']['angular'].append(end_state['angular'])
        
        joint_state = self.get_joint_state()
        self.data_buffer['state']['joint']['position'].append(joint_state['position'])
        self.data_buffer['state']['joint']['velocity'].append(joint_state['velocity'])
        self.data_buffer['state']['joint']['effort'].append(joint_state['effort'])
        
        robot_state = self.get_robot_state()
        self.data_buffer['state']['robot']['position'].append(robot_state['position'])
        self.data_buffer['state']['robot']['orientation'].append(robot_state['orientation'])
        
        # Generate and log action data
        actions = self.generate_action_commands()
        
        # Initialize action data for this timestep (empty if no actions)
        for action_type in ['effector', 'end', 'joint', 'robot']:
            if action_type in actions:
                for key, value in actions[action_type].items():
                    if key != 'index':
                        self.data_buffer['action'][action_type][key].append(value)
                    else:
                        self.data_buffer['action'][action_type][key].extend(value)
    
    def save_to_hdf5(self) -> None:
        """Save buffered data to HDF5 file."""
        with h5py.File(self.filename, 'w') as f:
            # Save timestamps
            f.create_dataset('timestamp', data=np.array(self.data_buffer['timestamp']))
            
            # Save state data
            state_group = f.create_group('state')
            
            # Effector state
            effector_group = state_group.create_group('effector')
            effector_group.create_dataset('position', data=np.array(self.data_buffer['state']['effector']['position']))
            
            # End effector state
            end_group = state_group.create_group('end')
            end_group.create_dataset('position', data=np.array(self.data_buffer['state']['end']['position']))
            end_group.create_dataset('orientation', data=np.array(self.data_buffer['state']['end']['orientation']))
            end_group.create_dataset('velocity', data=np.array(self.data_buffer['state']['end']['velocity']))
            end_group.create_dataset('angular', data=np.array(self.data_buffer['state']['end']['angular']))
            
            # Joint state
            joint_group = state_group.create_group('joint')
            joint_group.create_dataset('position', data=np.array(self.data_buffer['state']['joint']['position']))
            joint_group.create_dataset('velocity', data=np.array(self.data_buffer['state']['joint']['velocity']))
            joint_group.create_dataset('effort', data=np.array(self.data_buffer['state']['joint']['effort']))
            
            # Robot state
            robot_group = state_group.create_group('robot')
            robot_group.create_dataset('position', data=np.array(self.data_buffer['state']['robot']['position']))
            robot_group.create_dataset('orientation', data=np.array(self.data_buffer['state']['robot']['orientation']))
            
            # Save action data
            action_group = f.create_group('action')
            
            # Save action data for each component (only if data exists)
            for action_type in ['effector', 'end', 'joint', 'robot']:
                if any(self.data_buffer['action'][action_type].values()):
                    type_group = action_group.create_group(action_type)
                    for key, values in self.data_buffer['action'][action_type].items():
                        if values:  # Only save if there's data
                            type_group.create_dataset(key, data=np.array(values))
        
        # Update task info with completion data
        if len(self.data_buffer['timestamp']) > 1:
            duration = self.data_buffer['timestamp'][-1] - self.data_buffer['timestamp'][0]
            timesteps = len(self.data_buffer['timestamp'])
            self.update_task_info(duration, timesteps)
        
        print(f"Proprioceptive data saved to {self.filename}")
        print(f"Data structure created at: {self.base_data_path}")
    
    def print_data_summary(self) -> None:
        """Print a summary of collected data."""
        num_timesteps = len(self.data_buffer['timestamp'])
        num_actions = self.action_counter
        
        print(f"\nData Collection Summary:")
        print(f"- Total timesteps: {num_timesteps}")
        print(f"- Total actions sent: {num_actions}")
        print(f"- Data collection duration: {self.data_buffer['timestamp'][-1] - self.data_buffer['timestamp'][0]:.2f} seconds")
        
        # Print action breakdown
        for action_type in ['effector', 'end', 'joint', 'robot']:
            indices = self.data_buffer['action'][action_type].get('index', [])
            if indices:
                print(f"- {action_type.capitalize()} actions: {len(indices)}")


def main():
    """
    Main function to demonstrate proprioceptive data logging.
    """
    print("Starting proprioceptive data logging...")
    
    # Initialize logger with task ID and auto-increment episode ID
    logger = ProprioceptiveDataLogger(task_id="001", episode_id="auto")
    
    # Log data for 100 timesteps (about 10 seconds at 10Hz)
    num_steps = 100
    frequency = 10  # Hz
    dt = 1.0 / frequency
    
    print(f"Logging {num_steps} timesteps at {frequency}Hz...")
    print(f"Task ID: {logger.task_id}, Episode ID: {logger.episode_id}")
    print(f"Base data path: {logger.base_data_path}")
    
    for i in range(num_steps):
        logger.log_step()
        
        # Print progress every 20 steps
        if (i + 1) % 20 == 0:
            print(f"Logged {i + 1}/{num_steps} timesteps...")
        
        # Sleep to maintain frequency
        time.sleep(dt)
    
    # Save data to file
    logger.save_to_hdf5()
    
    # Print summary
    logger.print_data_summary()
    
    print("\nData logging completed!")
    
    # Print directory structure
    print("\nCreated directory structure:")
    print_directory_tree(logger.base_data_path)


def print_directory_tree(path: Path, prefix: str = "", max_depth: int = 4, current_depth: int = 0) -> None:
    """Print a directory tree structure."""
    if current_depth > max_depth:
        return
        
    if path.is_dir():
        print(f"{prefix}{path.name}/")
        if current_depth < max_depth:
            children = sorted(path.iterdir())
            for i, child in enumerate(children):
                is_last = i == len(children) - 1
                child_prefix = prefix + ("└── " if is_last else "├── ")
                next_prefix = prefix + ("    " if is_last else "│   ")
                
                if child.is_dir():
                    print(f"{child_prefix}{child.name}/")
                    print_directory_tree(child, next_prefix, max_depth, current_depth + 1)
                else:
                    print(f"{child_prefix}{child.name}")
    else:
        print(f"{prefix}{path.name}")


def inspect_hdf5_file(filename: str = None, task_id: str = "001", episode_id: str = "auto"):
    """
    Function to inspect the structure of the generated HDF5 file.
    """
    if filename is None:
        base_path = Path("/home/simt-wj/01_Active/smolvla_ws/data_atsume/data")
        
        if episode_id == "auto":
            # Find the latest episode
            proprio_task_dir = base_path / "proprio_stats" / task_id
            if proprio_task_dir.exists():
                episodes = [d.name for d in proprio_task_dir.iterdir() if d.is_dir()]
                if episodes:
                    # Sort episodes and get the latest one
                    episodes.sort()
                    episode_id = episodes[-1]
                else:
                    episode_id = "001"
            else:
                episode_id = "001"
        
        filename = base_path / "proprio_stats" / task_id / episode_id / "proprio_stats.h5"
    
    def print_structure(name, obj):
        print(name)
        if isinstance(obj, h5py.Dataset):
            print(f"  Shape: {obj.shape}, Dtype: {obj.dtype}")
    
    print(f"\nInspecting HDF5 file: {filename}")
    print("=" * 50)
    
    try:
        with h5py.File(filename, 'r') as f:
            f.visititems(print_structure)
    except FileNotFoundError:
        print(f"File {filename} not found. Run main() first to generate the data.")


def demo_multiple_episodes():
    """
    Demo function to show how to create multiple episodes automatically.
    """
    print("Demo: Creating multiple episodes automatically...")
    
    task_id = "001"
    
    for i in range(3):
        print(f"\n--- Creating Episode {i+1} ---")
        
        # Create logger with auto-increment
        logger = ProprioceptiveDataLogger(task_id=task_id, episode_id="auto")
        print(f"Auto-assigned Episode ID: {logger.episode_id}")
        
        # Log 20 timesteps for demo (faster)
        for j in range(20):
            logger.log_step()
        
        # Save data
        logger.save_to_hdf5()
        logger.print_data_summary()
    
    print("\nDemo completed! Check the data directory structure.")
    base_path = Path("/home/simt-wj/01_Active/smolvla_ws/data_atsume/data")
    print_directory_tree(base_path)


if __name__ == "__main__":
    main()
    
    # Optional: inspect the generated file
    print("\n" + "=" * 50)
    inspect_hdf5_file()
    
    # Uncomment the line below to run the multiple episodes demo
    # print("\n" + "=" * 50)
    # demo_multiple_episodes()

