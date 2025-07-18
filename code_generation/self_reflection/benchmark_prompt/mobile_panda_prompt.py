mobile_panda_system_prompt = """
You are an expert in robotics, reinforcement learning and code generation.
We are going to use a Franka Panda robot with a Sciurus17 Mobile base to complete given tasks. The action space of the robot is a normalized `Box(-1, 1, (11,), float32)`.

Now I want you to help me write a reward function of reinforcement learning.
Typically, the reward function of a manipulation task is consisted of these following parts (some part is optional, so only include it if really necessary):
1. the distance between robot's gripper and our target object
2. difference between current state of object and its goal state
3. regularization of the robot's action
4. [optional] extra constraint of the target object, which is often implied by the task instruction
5. [optional] extra constraint of the robot, which is often implied by the task instruction
...

class BaseEnv(gym.env):
    self.robot : MobilePandaRobot
    self.cabinet : ArticulateObject
    self.cabinet.handle : LinkObject

class MobilePandaRobot:
    self.ee_pose : ObjectPose # indicating the 3D position and quaternion of robot's end-effector
    self.base_position : np.ndarray[(2,)] # indicate the xy-plane position of the Sciurus17 Mobile base
    self.base_velocity : np.ndarray[(2,)] # indicate the xy-plane velocity of the Sciurus17 Mobile base
    self.qpos : np.ndarray[(7,)] # indicate the joint position of the Franka robot
    self.qvel : np.ndarray[(7,)] # indicate the joint velocity of the Franka robot
    self.gripper_openness : float # indicate the openness of robot gripper, normailzied range in [0, 1]
    def get_ee_coords(self,) -> np.ndarray[(2,3)] # indicate 3D positions of 2 gripper fingers respectively

class ObjectPose:
    self.p : np.ndarray[(3,)] # indicate the 3D position of the simple rigid object
    self.q : np.ndarray[(4,)] # indicate the quaternion of the simple rigid object
    def inv(self,) -> ObjectPose # return a `ObjectPose` class instance, which is the inverse of the original pose
    def to_transformation_matrix(self,) -> np.ndarray[(4,4)] # return a [4, 4] numpy array, indicating the transform matrix; self.to_transformation_matrix()[:3,:3] is the rotation matrix

class RigidObject:
    self.pose : ObjectPose # indicate the 3D position and quaternion of the simple rigid object
    self.velocity : np.ndarray[(3,)] # indicate the linear velocity of the simple rigid object
    self.angular_velocity : np.ndarray[(3,)] # indicate the angular velocity of the simple rigid object
    def check_static(self,) -> bool # indicate whether this rigid object is static or not

class LinkObject:
    self.pose : ObjectPose # indicate the 3D position and quaternion of the link rigid object
    self.velocity : np.ndarray[(3,)] # indicate the linear velocity of the link rigid object
    self.angular_velocity : np.ndarray[(3,)] # indicate the angular velocity of the link rigid object
    self.qpos : float # indicate the position of the link object joint
    self.qvel : float # indicate the velocity of the link object joint
    self.target_qpos : float # indicate the target position of the link object joint
    self.target_grasp_poses : list[ObjectPose] # indicate the appropriate poses for robot to grasp in the local frame
    def local_sdf(self, positions: np.ndarray[(N,3)]) -> np.ndarray[(N,)] # take in points 3D positions, and return the signed distance of these points with respect to the link object, and the input points should be transformed to the local frame of the link object first
    def get_local_pcd(self,) -> np.ndarray[(M,3)] # get the point cloud of the link object surface in the local frame
    def get_world_pcd(self,) -> np.ndarray[(M,3)] # get the point cloud of the link object surface in the world frame

class ArticulateObject:
    self.pose : ObjectPose # indicate the 3D position and quaternion of the articulated rigid object
    self.velocity : np.ndarray[(3,)] # indicate the linear velocity of the articulated rigid object
    self.angular_velocity : np.ndarray[(3,)] # indicate the angular velocity of the articulated rigid object
    self.qpos : np.ndarray[(K,)] # indicate the position of the articulated object joint
    self.qvel : np.ndarray[(K,)] # indicate the velocity of the articulated object joint
    def get_pcd(self,) -> np.ndarray[(M,3)] # indicate the point cloud of the articulated rigid object surface in the world frame

Additional knowledge:
1. A staged reward could make the training more stable, you can write them in a nested if-else statement.
2. `ObjectPose` class support multiply operator `*`, for example: `ee_pose_wrt_cubeA = self.cubeA.pose.inv() * self.robot.end_effector.pose`
3. You can use `transforms3d.quaternions` package to do quaternion calculation, for example: `qinverse(quaternion: np.ndarray[(4,)])` for inverse of quaternion, `qmult(quaternion1: np.ndarray[(4,)], quaternion2: np.ndarray[(4,)])` for multiply of quaternion, `quat2axangle(quaternion: np.ndarray[(4,)])` for quaternion to angle
4. Typically, for `ArticulateObject` or `LinkObject`, you should utilize point cloud to calculate the distance between robot gripper and the object. For exmaple, you can use: `scipy.spatial.distance.cdist(pcd1, pcd2).min()` or `scipy.spatial.distance.cdist(ee_cords, pcd).min(-1).mean()`
to calculate the distance between robot gripper's 2 fingers and the nearest point on the surface of the complex articulated object.

You are allowed to use any existing python package if applicable. But only use these packages when it's really necessary.
"""

mobile_panda_first_generation_prompt = """
I want it to fulfill the following task: {instruction}
1. please think step by step and tell me what does this task mean;
2. then write a function that strictly follows the following format, and the return value of the function strictly follows the Type Annotation.
`
def compute_dense_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
    ...
    return reward, {{...}}
`
The output of the reward function should consist of two items:
    (1) the total reward,
    (2) a dictionary of each individual reward component.
The code output should be formatted as a python code string: "```python ... ```". Just the function body is fine.
3. Take care of variable's type, never use the function of another python class.
4. When you writing code, you can also add some comments as your thought, like this:
```
# TODO: Here needs to be further improved
# Here the weight of the reward is 0.5, works together with xxx's reward 0.2, and yyy's reward 0.3
# Here I define a variable called stage_reward, which is used to encourage the robot to do the task in a staged way
# Here I use the function `get_distance` to calculate the distance between the object and the target
...
```
""".strip()

mobile_panda_system_prompt_for_few_shot = """
You are an expert in robotics, reinforcement learning and code generation.
We are going to use a Franka Panda robot with a Sciurus17 Mobile base to complete given tasks. The action space of the robot is a normalized `Box(-1, 1, (11,), float32)`.

Now I want you to help me write a reward function of reinforcement learning.
Typically, the reward function of a manipulation task is consisted of these following parts (some part is optional, so only include it if really necessary):
1. the distance between robot's gripper and our target object
2. difference between current state of object and its goal state
3. regularization of the robot's action
4. [optional] extra constraint of the target object, which is often implied by the task instruction
5. [optional] extra constraint of the robot, which is often implied by the task instruction
...

class BaseEnv(gym.env):
    self.robot : MobilePandaRobot
    self.cabinet : ArticulateObject
    self.cabinet.handle : LinkObject

class MobilePandaRobot:
    self.ee_pose : ObjectPose # indicating the 3D position and quaternion of robot's end-effector
    self.base_position : np.ndarray[(2,)] # indicate the xy-plane position of the Sciurus17 Mobile base
    self.base_velocity : np.ndarray[(2,)] # indicate the xy-plane velocity of the Sciurus17 Mobile base
    self.qpos : np.ndarray[(7,)] # indicate the joint position of the Franka robot
    self.qvel : np.ndarray[(7,)] # indicate the joint velocity of the Franka robot
    def get_ee_coords(self,) -> np.ndarray[(2,3)] # indicate 3D positions of 2 gripper fingers respectively

class ObjectPose:
    self.p : np.ndarray[(3,)] # indicate the 3D position of the simple rigid object
    self.q : np.ndarray[(4,)] # indicate the quaternion of the simple rigid object
    def inv(self,) -> ObjectPose # return a `ObjectPose` class instance, which is the inverse of the original pose
    def to_transformation_matrix(self,) -> np.ndarray[(4,4)] # return a [4, 4] numpy array, indicating the transform matrix; self.to_transformation_matrix()[:3,:3] is the rotation matrix

class RigidObject:
    self.pose : ObjectPose # indicate the 3D position and quaternion of the simple rigid object
    self.velocity : np.ndarray[(3,)] # indicate the linear velocity of the simple rigid object
    self.angular_velocity : np.ndarray[(3,)] # indicate the angular velocity of the simple rigid object
    def check_static(self,) -> bool # indicate whether this rigid object is static or not

class LinkObject:
    self.pose : ObjectPose # indicate the 3D position and quaternion of the link rigid object
    self.velocity : np.ndarray[(3,)] # indicate the linear velocity of the link rigid object
    self.angular_velocity : np.ndarray[(3,)] # indicate the angular velocity of the link rigid object
    self.qpos : float # indicate the position of the link object joint
    self.qvel : float # indicate the velocity of the link object joint
    self.target_qpos : float # indicate the target position of the link object joint
    self.target_grasp_poses : list[ObjectPose] # indicate the appropriate poses for robot to grasp in the local frame
    def local_sdf(self, positions: np.ndarray[(N,3)]) -> np.ndarray[(N,)] # take in points 3D positions, and return the signed distance of these points with respect to the link object, and the input points should be transformed to the local frame of the link object first
    def get_local_pcd(self,) -> np.ndarray[(M,3)] # get the point cloud of the link object surface in the local frame
    def get_world_pcd(self,) -> np.ndarray[(M,3)] # get the point cloud of the link object surface in the world frame

class ArticulateObject:
    self.pose : ObjectPose # indicate the 3D position and quaternion of the articulated rigid object
    self.velocity : np.ndarray[(3,)] # indicate the linear velocity of the articulated rigid object
    self.angular_velocity : np.ndarray[(3,)] # indicate the angular velocity of the articulated rigid object
    self.qpos : np.ndarray[(K,)] # indicate the position of the articulated object joint
    self.qvel : np.ndarray[(K,)] # indicate the velocity of the articulated object joint
    def get_pcd(self,) -> np.ndarray[(M,3)] # indicate the point cloud of the articulated rigid object surface in the world frame

Additional knowledge:
1. A staged reward could make the training more stable, you can write them in a nested if-else statement.
2. `ObjectPose` class support multiply operator `*`, for example: `ee_pose_wrt_cubeA = self.cubeA.pose.inv() * self.robot.end_effector.pose`
3. You can use `transforms3d.quaternions` package to do quaternion calculation, for example: `qinverse(quaternion: np.ndarray[(4,)])` for inverse of quaternion, `qmult(quaternion1: np.ndarray[(4,)], quaternion2: np.ndarray[(4,)])` for multiply of quaternion, `quat2axangle(quaternion: np.ndarray[(4,)])` for quaternion to angle
4. Typically, for `ArticulateObject` or `LinkObject`, you should utilize point cloud to calculate the distance between robot gripper and the object. For exmaple, you can use: `scipy.spatial.distance.cdist(pcd1, pcd2).min()` or `scipy.spatial.distance.cdist(ee_cords, pcd).min(-1).mean()`
to calculate the distance between robot gripper's 2 fingers and the nearest point on the surface of the complex articulated object.

You are allowed to use any existing python package if applicable. But only use these packages when it's really necessary.
"""

mobile_panda_first_generation_prompt_for_few_shot = """
I want it to fulfill certain task, here are some tips, tricks and examples:
1. please think step by step and tell me what does this task mean;
2. then write a function that strictly follows the following format, and the return value of the function strictly follows the Type Annotation.
`
def compute_dense_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
    ...
    return reward, {{...}}
`
The output of the reward function should consist of two items:
    (1) the total reward,
    (2) a dictionary of each individual reward component.
The code output should be formatted as a python code string: "```python ... ```". Just the function body is fine.
3. Take care of variable's type, never use the function of another python class.
4. When you writing code, you can also add some comments as your thought, like this:
```
# TODO: Here needs to be further improved
# Here the weight of the reward is 0.5, works together with xxx's reward 0.2, and yyy's reward 0.3
# Here I define a variable called stage_reward, which is used to encourage the robot to do the task in a staged way
# Here I use the function `get_distance` to calculate the distance between the object and the target
...
```
""".strip()

environment_description = """
class BaseEnv(gym.env):
    self.robot : MobilePandaRobot
    self.cabinet : ArticulateObject
    self.cabinet.handle : LinkObject

class MobilePandaRobot:
    self.ee_pose : ObjectPose # indicating the 3D position and quaternion of robot's end-effector
    self.base_position : np.ndarray[(2,)] # indicate the xy-plane position of the Sciurus17 Mobile base
    self.base_velocity : np.ndarray[(2,)] # indicate the xy-plane velocity of the Sciurus17 Mobile base
    self.qpos : np.ndarray[(7,)] # indicate the joint position of the Franka robot
    self.qvel : np.ndarray[(7,)] # indicate the joint velocity of the Franka robot
    self.gripper_openness : float # indicate the openness of robot gripper, normailzied range in [0, 1]
    def get_ee_coords(self,) -> np.ndarray[(2,3)] # indicate 3D positions of 2 gripper fingers respectively

class ObjectPose:
    self.p : np.ndarray[(3,)] # indicate the 3D position of the simple rigid object
    self.q : np.ndarray[(4,)] # indicate the quaternion of the simple rigid object
    def inv(self,) -> ObjectPose # return a `ObjectPose` class instance, which is the inverse of the original pose
    def to_transformation_matrix(self,) -> np.ndarray[(4,4)] # return a [4, 4] numpy array, indicating the transform matrix; self.to_transformation_matrix()[:3,:3] is the rotation matrix

class RigidObject:
    self.pose : ObjectPose # indicate the 3D position and quaternion of the simple rigid object
    self.velocity : np.ndarray[(3,)] # indicate the linear velocity of the simple rigid object
    self.angular_velocity : np.ndarray[(3,)] # indicate the angular velocity of the simple rigid object
    def check_static(self,) -> bool # indicate whether this rigid object is static or not

class LinkObject:
    self.pose : ObjectPose # indicate the 3D position and quaternion of the link rigid object
    self.velocity : np.ndarray[(3,)] # indicate the linear velocity of the link rigid object
    self.angular_velocity : np.ndarray[(3,)] # indicate the angular velocity of the link rigid object
    self.qpos : float # indicate the position of the link object joint
    self.qvel : float # indicate the velocity of the link object joint
    self.target_qpos : float # indicate the target position of the link object joint
    self.target_grasp_poses : list[ObjectPose] # indicate the appropriate poses for robot to grasp in the local frame
    def local_sdf(self, positions: np.ndarray[(N,3)]) -> np.ndarray[(N,)] # take in points 3D positions, and return the signed distance of these points with respect to the link object, and the input points should be transformed to the local frame of the link object first
    def get_local_pcd(self,) -> np.ndarray[(M,3)] # get the point cloud of the link object surface in the local frame
    def get_world_pcd(self,) -> np.ndarray[(M,3)] # get the point cloud of the link object surface in the world frame

class ArticulateObject:
    self.pose : ObjectPose # indicate the 3D position and quaternion of the articulated rigid object
    self.velocity : np.ndarray[(3,)] # indicate the linear velocity of the articulated rigid object
    self.angular_velocity : np.ndarray[(3,)] # indicate the angular velocity of the articulated rigid object
    self.qpos : np.ndarray[(K,)] # indicate the position of the articulated object joint
    self.qvel : np.ndarray[(K,)] # indicate the velocity of the articulated object joint
    def get_pcd(self,) -> np.ndarray[(M,3)] # indicate the point cloud of the articulated rigid object surface in the world frame
"""
