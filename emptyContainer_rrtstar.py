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

#desired_ee_rp = fr.ee(fr.home_joints)[3:5]
def ee_upright_constraint(q):
    '''
    TODO: Implement constraint function and its gradient. 
    
    This constraint should enforce the end-effector stays upright.
    Hint: Use the roll and pitch angle in desired_ee_rp. The end-effector is upright in its home state.

    Input:
        q - a joint configuration

    Output:
        err - a non-negative scalar that is 0 when the constraint is satisfied
        grad - a vector of length 6, where the ith element is the derivative of err w.r.t. the ith element of ee
    '''
    ee = fr.ee(q)
    err, grad = None, None

    grad = np.zeros(6)
    err = (ee[3] - desired_ee_rp[0])*(ee[3] - desired_ee_rp[0]) + (ee[4] - desired_ee_rp[1])*(ee[4] - desired_ee_rp[1])
    # error function should be convex
    grad = [0, 0, 0, 2*(ee[3] - desired_ee_rp[0]), 2*(ee[4] - desired_ee_rp[1]), 0]
    
    return err, grad


if __name__ == "__main__":
	action_mode = ActionMode(ArmActionMode.ABS_EE_POSE) # See rlbench/action_modes.py for other action modes
	env = Environment(action_mode, '', ObservationConfig(), False)
	task = env.get_task(EmptyContainer) # available tasks: EmptyContainer, PlayJenga, PutGroceriesInCupboard, SetTheTable
	agent = RandomAgent()
	obj_pose_sensor = NoisyObjectPoseSensor(env)

	descriptions, obs = task.reset()
	print(descriptions)

	fr = FrankaRobot()
	rrt = RRT(fr, None)

	arm = env._scene._active_task.robot.arm
	# Go to location above the center container (waypoint 2)
	curr_pos = obs.gripper_pose
	# IK on it
	constraint = None#ee_upright_constraint

	#w,x,y,z = curr_pos[3:]

	# joints_start = arm.get_configs_for_tip_pose(
 #                                 curr_pos[0:3],
 #                                 None,
 #                                 [x,y,z,w],
 #                                 ignore_collisions=True
 #                                 )
	#print("start", joints_start)
	# joints_start = arm.solve_ik(
	#                                  curr_pos[0:3],
	#                                  None,
	#                                  curr_pos[3:],
	#                                  )

	# Try to pick up shape 0
	# inter1 = curr_pos.copy()
	# inter1[2] = inter1[2] - 0.30
	# # w,x,y,z = inter1[3:]

	# # joints_inter1 = arm.get_configs_for_tip_pose(
 # #                                 inter1[0:3],
 # #                                 None,
 # #                                 [x,y,z,w],
 # #                                 ignore_collisions=True
 # #                                 )

	# plan = rrt.plan(np.asarray(curr_pos), np.asarray(inter1), constraint)

	# for p in plan:
	# 	joints = p
	# 	task.step(p.tolist()+[1])

	waypoint2 = obj_pose_sensor.get_poses()["waypoint2"]

	# w,x,y,z = waypoint2[3:]
	# # IK on it
	# joints_target = arm.get_configs_for_tip_pose(
 #                                 waypoint2[0:3],
 #                                 None,
 #                                 [x,y,z,w],
 #                                 ignore_collisions=True
 #                                 )
	# #print("target", joints_target)
	# # joints_target = arm.solve_ik(
	# #                                  shape0[0:3],
	# #                                  None,
	# #                                  shape0[3:],
	# #                                  )
	
	# desired_ee_rp = fr.ee(fr.home_joints)[3:5]
	plan = rrt.plan(np.asarray(curr_pos), np.asarray(waypoint2), constraint)

	#print("plan = ", plan)

	for p in plan:
		joints = p
		task.step(p.tolist()+[1])
		


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