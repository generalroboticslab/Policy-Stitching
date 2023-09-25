from typing import Any, Dict, Union

import numpy as np

from my_panda_gym.envs.core import Task
from my_panda_gym.utils import distance


class Reach3(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_range=0.3,  # 0.3 0.05
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        # self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        # self.sim.create_sphere(
        #     body_name="target",
        #     radius=0.02,
        #     mass=0.0,
        #     ghost=True,
        #     # position=np.zeros(3),
        #     position=np.array([0.0, 0.0, 0.02]),
        #     rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        # )

        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * 0.02,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.02]),
            rgba_color=np.array([0.1, 0.1, 0.9, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        # there is no position, rotation of the object
        # but I set them all zeros to take the places in obs
        # so the task obs of all the tasks have the same length
        # object_position = np.random.uniform(low=-0.01, high=0.01, size=3)
        object_position = np.array([0.0, 0.0, 0.02]) + self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        object_rotation = np.random.uniform(low=-0.01, high=0.01, size=3)
        object_velocity = np.random.uniform(low=-0.001, high=0.001, size=3)
        object_angular_velocity = np.random.uniform(low=-0.001, high=0.001, size=3)
        # object_position = np.zeros(3)
        # object_rotation = np.zeros(3)
        # object_velocity = np.zeros(3)
        # object_angular_velocity = np.zeros(3)
        observation = np.concatenate(
            [
                object_position,
                object_rotation,
                object_velocity,
                object_angular_velocity,
            ]
        )
        return observation  # no tasak-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = np.array([0.0, 0.0, 0.02])  # z offset for the goal sphere
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        goal += noise
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float64)
        else:
            return -d
