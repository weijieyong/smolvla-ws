refer from https://huggingface.co/datasets/agibot-world/AgiBotDigitalWorld

data_path/
├── task_info/
│   └── task_{task_id}.json
├── observations/
│   └── {task_id}/
│       └── {episode_id}/
│           ├── depth/
│           │   ├── hand_cam_depth_000001.png
│           │   ├── side_cam_depth_000002.png
│           │   └── ...
│           └── videos/
│               ├── hand_cam_1.mp4
│               ├── side_cam_2.mp4
│               └── top_cam_3.mp4
└── proprio_stats/
    └── {task_id}/
        └── {episode_id}/
            └── proprio_stats.h5


inside `task_info.json`
store the basic information of every episode, language instructions



inside `proprio_stats.h5`

|-- timestamp
|-- state
    |-- effector
        |-- force
        |-- index
        |-- position
    |-- end
        |-- angular
        |-- orientation
        |-- position
        |-- velocity
        |-- wrench
    |-- joint
        |-- current_value
        |-- effort
        |-- position
        |-- velocity
    |-- robot
        |-- orientation
        |-- orientation_drift
        |-- position
        |-- position_drift
|-- action
    |-- effector
        |-- force
        |-- index
        |-- position
    |-- end
        |-- angular
        |-- orientation
        |-- position
        |-- velocity
        |-- wrench
    |-- joint
        |-- effort
        |-- index
        |-- position
        |-- velocity
    |-- robot
        |-- index
        |-- orientation
        |-- position
        |-- velocity

## Explanation of Proprioceptive State
### Terminology
*The definitions and data ranges in this section may change with software and hardware version. Stay tuned.*

**State and action**
1. State
State refers to the monitoring information of different sensors and actuators.
2. Action
Action refers to the instructions sent to the hardware abstraction layer, where controller would respond to these instructions. Therefore, there is a difference between the issued instructions and the actual executed state.

**Actuators**
1. ***Effector:*** refers to the end effector, for example dexterous hands or grippers.
2. ***End:*** refers to the 6DoF end pose on the robot flange.
4. ***Joint:*** refers to the joints of the robot, with 34 degrees of freedom (2 DoF head, 2 Dof waist, 7 DoF each arm, 8 Dof each gripper).
5. ***Robot:*** refers to the robot's pose in its surrouding environment. The orientation and position refer to the robot's relative pose in the odometry coordinate syste

### Common fields
1. Position: Spatial position, encoder position, angle, etc.
2. Velocity: Speed
3. Angular: Angular velocity
4. Effort: Torque of the motor. Not available for now.
5. Wrench: Six-dimensional force, force in the xyz directions, and torque. Not available for now.

### Value shapes and ranges
| Group | Shape | Meaning | 
| --- | :---- | :---- |
| /timestamp | [N] | timestamp in seconds:nanoseconds in simulation time |
| /state/effector/position (gripper) | [N, 2] | left `[:, 0]`, right `[:, 1]`, gripper open range in mm |
| /state/end/orientation | [N, 2, 4] | left `[:, 0, :]`, right `[:, 1, :]`, flange quaternion with wxyz |
| /state/end/position | [N, 2, 3] | left `[:, 0, :]`, right `[:, 1, :]`, flange xyz in meters |
| /state/joint/position | [N, 34] | joint position based on joint names | 
| /state/joint/velocity | [N, 34] | joint velocity based on joint names | 
| /state/joint/effort | [N, 34] | joint effort based on joint names | 
| /state/robot/orientation | [N, 4] | quaternion in wxyz |
| /state/robot/position | [N, 3] | xyz position, where z is always 0 in meters |
| /action/*/index | [M] | actions indexes refer to when the control source is actually sending signals |
| /action/effector/position (gripper) | [N, 2] | left `[:, 0]`, right `[:, 1]`, gripper open range in mm |
| /action/end/orientation | [N, 2, 4] | same as /state/end/orientation |
| /action/end/position | [N, 2, 3] | same as /state/end/position |
| /action/end/index | [M_2] | same as other indexes |
| /action/joint/position | [N, 14] | same as /state/joint/position |
| /action/joint/index | [M_4] | same as other indexes |
| /action/robot/velocity | [N, 2] | vel along x axis `[:, 0]`, yaw rate `[:, 1]` |
| /action/robot/index | [M_5] | same as other indexes | 