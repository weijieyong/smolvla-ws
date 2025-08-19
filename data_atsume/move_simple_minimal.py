#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import threading
import sys
import logging
from pathlib import Path
import numpy as np

# Add parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from realman_lib.robotic_arm import Arm, RM65

# Configuration
ROBOT_IP = "192.168.1.18"
ROBOT_MODEL = RM65
LOGGING_RATE_HZ = 50
SPEED_PERCENT = 10

# Global variables for threading
logging_active = False
log_data = []

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def joint_position_logger(arm: Arm):
    """Log joint positions at 50Hz in a separate thread."""
    global logging_active, log_data
    
    logger.info("Started joint position logging at 50Hz")
    sleep_time = 1.0 / LOGGING_RATE_HZ  # 20ms for 50Hz
    
    while logging_active:
        try:
            ret_code, joints = arm.Get_Joint_Degree()
            if ret_code == 0 and joints:
                timestamp_ns = time.time_ns()  # Get timestamp in nanoseconds
                log_entry = {
                    'timestamp_ns': timestamp_ns,
                    'joints': joints.copy()
                }
                log_data.append(log_entry)
                
                # Print current joint positions (overwrite line)
                # joint_str = ", ".join([f"{j:7.2f}" for j in joints[:6]])
                # print(f"\rJoints: [{joint_str}]", end="", flush=True)
                
        except Exception as e:
            logger.warning(f"Logging error: {e}")
            
        time.sleep(sleep_time)
    
    print()  # New line after logging stops
    logger.info("Joint position logging stopped")


def movej_with_logging(arm: Arm, target_joints: list, description: str = ""):
    """Execute MoveJ command while logging joint positions."""
    global logging_active, log_data
    
    # if description:
    #     logger.info(f"Moving to {description}: {target_joints}")
    # else:
    #     logger.info(f"Moving to joints: {target_joints}")
    
    # Clear previous log data
    log_data.clear()
    
    # Start logging thread
    logging_active = True
    log_thread = threading.Thread(target=joint_position_logger, args=(arm,))
    log_thread.daemon = True
    log_thread.start()
    
    try:
        # Execute MoveJ command (non-blocking)
        arm.Movej_Cmd(joint=target_joints, v=SPEED_PERCENT, block=False)
        
        # Simple wait for movement completion
        # You could implement more sophisticated completion detection here
        time.sleep(3.0)  # Adjust based on your move distance/speed
        
    finally:
        # Stop logging
        logging_active = False
        log_thread.join(timeout=1.0)
        
        logger.info(f"Movement completed. Logged {len(log_data)} data points")
        
        # Save log data if needed
        if log_data:
            save_log_data(log_data, description)


def save_log_data(data: list, description: str = ""):
    """Save logged joint data to CSV file with nanosecond timestamps."""
    filename = f"joint_log_{int(time.time())}.csv"
    if description:
        filename = f"joint_log_{description.replace(' ', '_')}_{int(time.time())}.csv"
    
    try:
        with open(filename, 'w') as f:
            # Write CSV header
            f.write("timestamp_ns,joint1,joint2,joint3,joint4,joint5,joint6\n")
            
            # Write data rows
            for entry in data:
                timestamp_ns = entry['timestamp_ns']
                joints = entry['joints']
                joint_str = ",".join([f"{j:.6f}" for j in joints[:6]])
                f.write(f"{timestamp_ns},{joint_str}\n")
        
        logger.info(f"Saved {len(data)} data points to {filename}")
    except Exception as e:
        logger.error(f"Failed to save log data: {e}")


def main():
    """Main function with simple MoveJ movements."""
    arm = None
    try:
        # Initialize robot
        logger.info(f"Connecting to robot at {ROBOT_IP}...")
        arm = Arm(dev_mode=ROBOT_MODEL, ip=ROBOT_IP)
        
        if arm.Arm_Socket_State() != 0:
            raise SystemExit(f"Failed to connect. Error: {arm.Arm_Socket_State()}")
        
        logger.info("Connected successfully.")
        
        # Enable joints
        for i in range(1, 7):
            arm.Set_Joint_EN_State(joint_num=i, state=True)
            arm.Set_Joint_Err_Clear(joint_num=i)
        time.sleep(1)
        
        # Define some target positions
        positions = [
            ([0, -45, 90, 0, 90, 0], "home position"),
            ([10, -30, 75, 0, 75, 30], "position 1"),
            # ([-30, -60, 120, 0, 60, -30], "position 2"),
            ([0, -45, 90, 0, 90, 0], "back to home")
        ]
        
        # Execute movements
        for joints, desc in positions:
            movej_with_logging(arm, joints, desc)
            time.sleep(1)  # Pause between movements
            
        logger.info("All movements completed!")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        # Cleanup
        global logging_active
        logging_active = False
        
        if arm:
            arm.Move_Stop_Cmd()
            arm.Arm_Socket_Close()
            arm.RM_API_UnInit()
            logger.info("Disconnected from robot")


if __name__ == "__main__":
    main()
