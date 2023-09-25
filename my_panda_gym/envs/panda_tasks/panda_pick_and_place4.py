import numpy as np

from my_panda_gym.envs.core import RobotTaskEnv
from my_panda_gym.envs.robots.panda import Panda
from my_panda_gym.envs.tasks.pick_and_place4 import PickAndPlace4
from my_panda_gym.pybullet import PyBullet

from my_panda_gym.utils import distance
from typing import Any, Dict, Union


class PandaPickAndPlaceEnv4(RobotTaskEnv):
    """Pick and Place task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        sim = PyBullet(render=render)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = PickAndPlace4(sim, reward_type=reward_type)
        super().__init__(robot, task)



    # def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
    #     d = distance(achieved_goal, desired_goal)
    #     if self.reward_type == "sparse":
    #         return -np.array(d > self.distance_threshold, dtype=np.float64)
    #     elif self.reward_type == "shaping":
    #         ee_position = self.get_ee_position()
    #         d1 = distance(ee_position, achieved_goal)
    #         d2 = distance(achieved_goal, desired_goal)
    #         return -d1-d2
    #     else:
    #         return -d
