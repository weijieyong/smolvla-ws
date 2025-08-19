"""Minimal gripper logging at 30Hz with open/close cycle."""

import csv
import time
import threading
from pyrobotiqgripper import RobotiqGripper

# Simple lock to prevent simultaneous serial access
comm_lock = threading.Lock()


def log_gripper_data(gripper, output_path, running_flag):
    """Log gripper position at 30Hz."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ns", "position_mm", "object_detected"])
        
        while running_flag[0]:
            try:
                with comm_lock:
                    gripper.readAll()
                    pos_mm = gripper.getPositionmm()
                    obj_detected = gripper.paramDic.get("gOBJ", "")
                writer.writerow([time.time_ns(), pos_mm, obj_detected])
                f.flush()
            except:
                writer.writerow([time.time_ns(), "", ""])
                f.flush()
            
            time.sleep(1.0 / 30.0)  # 30Hz


def control_gripper(gripper, running_flag):
    """Simple open/close cycle with non-blocking movement."""
    while running_flag[0]:
        try:
            # Start closing movement (non-blocking)
            with comm_lock:
                gripper.write_registers(1000, [0b0000100100000000, 200, 100 * 0b100000000 + 100])
            
            # Wait for movement to complete while allowing position logging
            time.sleep(0.5)  # Give movement time to start
            for _ in range(50):  # Check for up to 5 seconds
                if not running_flag[0]:
                    break
                time.sleep(0.1)
                
                # Quick check if movement done (without blocking logging too much)
                try:
                    with comm_lock:
                        gripper.readAll()
                        if gripper.paramDic.get("gGTO", 1) == 0:  # Movement complete
                            break
                except:
                    pass

            time.sleep(1)  # Stay closed
            
            # Start opening movement (non-blocking)
            with comm_lock:
                gripper.write_registers(1000, [0b0000100100000000, 0, 100 * 0b100000000 + 100])
            
            # Wait for movement to complete while allowing position logging
            time.sleep(0.5)  # Give movement time to start
            for _ in range(50):  # Check for up to 5 seconds
                if not running_flag[0]:
                    break
                time.sleep(0.1)
                
                # Quick check if movement done
                try:
                    with comm_lock:
                        gripper.readAll()
                        if gripper.paramDic.get("gGTO", 1) == 0:  # Movement complete
                            break
                except:
                    pass
            
            time.sleep(1)  # Stay opened
            
        except Exception as e:
            print(f"Control error: {e}")
            time.sleep(1)


if __name__ == "__main__":
    g = RobotiqGripper()
    
    # Quick setup
    try:
        if not g.isActivated():
            g.activate()
        g.calibrate(2, 85)
        print("Gripper ready")
    except Exception as e:
        print(f"Setup failed: {e}")
    
    # Shared flag for stopping threads
    running = [True]
    
    # Start threads
    log_thread = threading.Thread(target=log_gripper_data, args=(g, "gripper_stream.csv", running))
    control_thread = threading.Thread(target=control_gripper, args=(g, running))
    
    log_thread.start()
    control_thread.start()
    
    print("Logging at 30Hz. Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        running[0] = False
        print("Stopped")
