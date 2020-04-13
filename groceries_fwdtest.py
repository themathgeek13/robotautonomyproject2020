import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *

from perception import CameraIntrinsics, DepthImage

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

if __name__ == "__main__":
	action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN) # See rlbench/action_modes.py for other action modes
	env = Environment(action_mode, '', ObservationConfig(), False)
	task = env.get_task(PutGroceriesInCupboard) # available tasks: EmptyContainer, PlayJenga, PutGroceriesInCupboard, SetTheTable
	agent = RandomAgent()
	obj_pose_sensor = NoisyObjectPoseSensor(env)

	descriptions, obs = task.reset()
	print(descriptions)

	# Set all the object poses (create function because it might change if you drop something)
	def coffee():
		return obj_pose_sensor.get_poses()["coffee_grasp_point"]

	def chocolate_jello():
		return obj_pose_sensor.get_poses()["chocolate_jello_grasp_point"]

	def crackers():
		return obj_pose_sensor.get_poses()["crackers_grasp_point"]

	def mustard():
		return obj_pose_sensor.get_poses()["mustard_grasp_point"]

	def soup():
		return obj_pose_sensor.get_poses()["soup_grasp_point"]

	def spam():
		return obj_pose_sensor.get_poses()["spam_grasp_point"]

	def strawberry_jello():
		return obj_pose_sensor.get_poses()["strawberry_jello_grasp_point"]

	# Set all the waypoints
	waypoint1 = obj_pose_sensor.get_poses()["waypoint1"]
	waypoint2 = obj_pose_sensor.get_poses()["waypoint2"]
	waypoint3 = obj_pose_sensor.get_poses()["waypoint3"]	# just outside cupboard
	waypoint4 = obj_pose_sensor.get_poses()["waypoint4"]	# lower shelf of cupboard
	startpose = obs.gripper_pose
	# Go to location above the center of workspace (waypoint 2)
	#task.step(waypoint2.tolist()+[1])

	# Now go to crackers
	task.step(crackers().tolist()+[1])
	# Now close the gripper to try picking it up (close gripper)
	task.step((obs.gripper_pose).tolist()+[0])

	# Now move to waypoint 2 then waypoint 3 once again (but keep the gripper closed)
	#task.step(waypoint2.tolist()+[0])
	task.step(waypoint3.tolist()+[0])
	task.step(waypoint4.tolist()+[0])

	# Now drop it.
	task.step((obs.gripper_pose).tolist()+[1])
	#######################################################################
	# Now move back to startpose
	task.step(waypoint3.tolist()+[1])
	#task.step(startpose.tolist()+[1])
	# Now pick up coffee
	task.step(coffee().tolist()+[1])
	# Now close the gripper to try picking it up (close gripper)
	task.step((obs.gripper_pose).tolist()+[0])

	# Now move to waypoint 2 once again (but keep the gripper closed)
	#task.step(waypoint2.tolist()+[0])
	task.step(waypoint3.tolist()+[0])
	task.step(waypoint4.tolist()+[0])

	# Now drop it.
	task.step((obs.gripper_pose).tolist()+[1])
	######################################################################
	# Now move back to waypoint3
	task.step(waypoint3.tolist()+[1])
	#task.step(startpose.tolist()+[1])
	# Now pick up the mustard
	task.step(mustard().tolist()+[1])
	# Now close the gripper to try picking it up (close gripper)
	task.step((obs.gripper_pose).tolist()+[0])

	# Now move to waypoint 2 once again (but keep the gripper closed)
	#task.step(waypoint2.tolist()+[0])
	task.step(waypoint3.tolist()+[0])
	task.step(waypoint4.tolist()+[0])

	# Now drop it.
	task.step((obs.gripper_pose).tolist()+[1])
	######################################################################
	# Now move back to waypoint3
	task.step(waypoint3.tolist()+[1])
	#task.step(startpose.tolist()+[1])

	# Now pick up the spam
	task.step(spam().tolist()+[1])
	# Now close the gripper to try picking it up (close gripper)
	task.step((obs.gripper_pose).tolist()+[0])

	# Now move to waypoint 2 once again (but keep the gripper closed)
	#task.step(waypoint2.tolist()+[0])
	task.step(waypoint3.tolist()+[0])
	task.step(waypoint4.tolist()+[0])

	# Now drop it.
	task.step((obs.gripper_pose).tolist()+[1])
	######################################################################
	# Now move back to waypoint3
	task.step(waypoint3.tolist()+[1])
	#task.step(startpose.tolist()+[1])

	# Now pick up the strawberry jello
	task.step(strawberry_jello().tolist()+[1])
	# Now close the gripper to try picking it up (close gripper)
	task.step((obs.gripper_pose).tolist()+[0])

	# Now move to waypoint 2 once again (but keep the gripper closed)
	#task.step(waypoint2.tolist()+[0])
	task.step(waypoint3.tolist()+[0])
	task.step(waypoint4.tolist()+[0])

	# Now drop it.
	task.step((obs.gripper_pose).tolist()+[1])

	env.shutdown()

