import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion

import sys
import os
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *

from perception import CameraIntrinsics, DepthImage
from scipy.spatial.transform import Rotation as R


from rrt import RRT
from franka_robot import FrankaRobot 

def skew(x):
    return np.array([[0, -x[2], x[1]],
                    [x[2], 0, -x[0]],
                    [-x[1], x[0], 0]])


def sample_normal_pose(pos_scale, rot_scale):
    '''
    Samples a 6D pose from a zero-mean isotropic normal distribution
    '''
    pos = np.random.normal(scale=pos_scale)
        
    eps = skew(np.random.normal(scale=rot_scale))
    R = sp.linalg.expm(eps)
    quat_wxyz = from_rotation_matrix(R)

    return pos, quat_wxyz


class RandomAgent:

    def act(self, obs):
        delta_pos = [(np.random.rand() * 2 - 1) * 0.005, 0, 0]
        delta_quat = [0, 0, 0, 1] # xyzw
        gripper_pos = [np.random.rand() > 0.5]
        return delta_pos + delta_quat + gripper_pos


class NoisyObjectPoseSensor:

    def __init__(self, env):
        self._env = env

        self._pos_scale = [0.005] * 3
        self._rot_scale = [0.01] * 3

    def get_poses(self):
        objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
        obj_poses = {}

        for obj in objs:
            name = obj.get_name()
            pose = obj.get_pose()

            pos, quat_wxyz = sample_normal_pose(self._pos_scale, self._rot_scale)
            gt_quat_wxyz = quaternion(pose[6], pose[3], pose[4], pose[5])
            perturbed_quat_wxyz = quat_wxyz * gt_quat_wxyz

            pose[:3] += pos
            pose[3:] = [perturbed_quat_wxyz.x, perturbed_quat_wxyz.y, perturbed_quat_wxyz.z, perturbed_quat_wxyz.w]

            obj_poses[name] = pose

        return obj_poses

def move_to_location(task, obs, desired_pose, obj_pose_sensor, tolerance=0.5, isopen=1):
	while True:
		obj_poses = obj_pose_sensor.get_poses()
		current_gripper_pose = obs.gripper_pose
		delta = desired_pose - current_gripper_pose 	# this is the entire delta. First servo with XYZ.
		delta[3:] = np.zeros(4)		# don't change the quaternion 
		print(np.linalg.norm(delta))
		if(np.linalg.norm(delta)<tolerance):
			break
		obs, reward, terminate = task.step((current_gripper_pose+delta/20).tolist()+[isopen])
	return obs, reward, terminate


def move_rrt(task, source, dest, open=1):
	plan = rrt.plan(np.asarray(source), np.asarray(dest), None)
	for p in plan:
		joints = p
		obs, reward, terminate = task.step(p.tolist()+[open])
	return obs, reward, terminate

def check_shape_loc(obs, shape_ind):
	objPoses = obj_pose_sensor.get_poses()
	
	if shape_ind == 0:
		upright_s0 = objPoses["Shape0"]
		upright_s0[3:] = obs.gripper_pose[3:]
		return upright_s0

	if shape_ind == 1:
		upright_s1 = objPoses["Shape1"]
		upright_s1[3:] = obs.gripper_pose[3:]
		return upright_s1
	
	if shape_ind == 3:
		upright_s3 = objPoses["Shape3"]
		upright_s3[3:] = obs.gripper_pose[3:]
		return upright_s3




if __name__ == "__main__":
	action_mode = ActionMode(ArmActionMode.ABS_EE_POSE) # See rlbench/action_modes.py for other action modes
	env = Environment(action_mode, '', ObservationConfig(), False, static_positions=True)
	task = env.get_task(EmptyContainer) # available tasks: EmptyContainer, PlayJenga, PutGroceriesInCupboard, SetTheTable
	agent = RandomAgent()
	obj_pose_sensor = NoisyObjectPoseSensor(env)

	descriptions, obs = task.reset()
	print(descriptions)

	fr = FrankaRobot()
	rrt = RRT(fr, None)
	
	arm = env._scene._active_task.robot.arm

	objPoses = obj_pose_sensor.get_poses()

	# First move the robot from initial pose to waypoint 2
	curr_pos = obs.gripper_pose
	
	waypoint2 = objPoses["waypoint2"]

	upright_w2 = waypoint2
	upright_w2[3:] = obs.gripper_pose[3:] 

	obs, reward, terminate = move_rrt(task, curr_pos, waypoint2)

	waypoint3 = objPoses["waypoint3"]
	upright_w3 = waypoint3
	upright_w3[3:] = obs.gripper_pose[3:] 

		
	print("Forward pass begins:")
	for i in [0, 1, 3]:
		shape_loc = check_shape_loc(obs, i)
		obs, reward, terminate = move_rrt(task, waypoint2, shape_loc)
		obs, reward, terminate = task.step((obs.gripper_pose).tolist()+[0])
		obs, reward, terminate = move_rrt(task, shape_loc, waypoint2, 0)
		obs, reward, terminate = move_rrt(task, waypoint2, waypoint3, 0)
		droploc = waypoint3.copy()
		droploc[2]-=0.1
		obs, reward, terminate = move_rrt(task, waypoint3, droploc, 0)
		obs, reward, terminate = task.step((obs.gripper_pose).tolist()+[1])
		obs, reward, terminate = move_rrt(task, droploc, waypoint3)
		obs, reward, terminate = move_rrt(task, waypoint3, waypoint2)

	
	print("Resetting begins:")

	r = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
	r2 = R.from_quat(obs.gripper_pose[3:])
	rot_gripper_pose = (r*r2).as_quat()
	finalpose = obs.gripper_pose
	finalpose[3:] = rot_gripper_pose

	# Rotate gripper by 90 degrees
	obs, reward, terminate = move_rrt(task, obs.gripper_pose, finalpose)

	# Get updated positions of waypoints (gripper rotated)
	waypoint2 = obj_pose_sensor.get_poses()["waypoint2"]
	upright_w2 = waypoint2
	upright_w2[3:] = obs.gripper_pose[3:]

	waypoint3 = obj_pose_sensor.get_poses()["waypoint3"]
	upright_w3 = waypoint3
	upright_w3[3:] = obs.gripper_pose[3:]


	for i in [0, 1, 3]:
		shape_loc = check_shape_loc(obs, i)
		obs, reward, terminate = move_rrt(task, waypoint2, waypoint3)
		obs, reward, terminate = move_rrt(task, waypoint3, shape_loc)
		obs, reward, terminate = task.step((obs.gripper_pose).tolist()+[0])
		obs, reward, terminate = move_rrt(task, shape_loc, waypoint3, 0)
		obs, reward, terminate = move_rrt(task, waypoint3, waypoint2, 0)
		obs, reward, terminate = task.step((obs.gripper_pose).tolist()+[1])

	
	"""
    while True:
        # Getting noisy object poses
        obj_poses = obj_pose_sensor.get_poses()
        # Getting various fields from obs
        current_joints = obs.joint_positions
        gripper_pose = obs.gripper_pose
        rgb = obs.wrist_rgb
        depth = obs.wrist_depth
        mask = obs.wrist_mask
        # Perform action and step simulation
        action = agent.act(obs)
        obs, reward, terminate = obs, reward, terminate = task.step(action)
        depth = obs.wrist_depth
        depth_image = DepthImage(depth, frame='world')
        cam_intr = CameraIntrinsics(fx=110.851251684, fy=110.851251684, cx=64, cy=64, frame='world', height=128, width=128)
        point_cloud = cam_intr.deproject(depth_image)
        # if terminate:
        #     break
	"""
	env.shutdown()