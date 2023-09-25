import time
import math
import random

import numpy as np
import pybullet as p
import pybullet_data

# from robot_env.utilities import Models, Camera
from robot_env.utilities import Models, Camera
from collections import namedtuple
# from attrdict import AttrDict
from tqdm import tqdm
from typing import Any, Dict, Iterator, Optional, Union

# this env is for pushing the cube on slippery (low friction)
class Ur5Push4:

    SIMULATION_STEP_DELAY = 1 / 240.

    def __init__(self, robot, models: Models, camera=None, vis=False) -> None:
        self.robot = robot
        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)
        self.camera = camera

        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.planeID = p.loadURDF("plane.urdf")

        # self.ballID = p.loadSoftBody("ball.obj", simFileName="ball.vtk", basePosition=[0, 0, -1], scale=0.5, mass=4,
        #                         useNeoHookean=1, NeoHookeanMu=400, NeoHookeanLambda=600, NeoHookeanDamping=0.001,
        #                         useSelfCollision=1, frictionCoeff=.5, collisionMargin=0.001)

        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        # custom sliders to tune parameters (name of the parameter,range,initial value)
        self.xin = p.addUserDebugParameter("x", -0.224, 0.224, 0)
        self.yin = p.addUserDebugParameter("y", -0.224, 0.224, 0)
        self.zin = p.addUserDebugParameter("z", 0, 1., 0.5)
        self.rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
        self.pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, np.pi/2)
        self.yawId = p.addUserDebugParameter("yaw", -np.pi/2, np.pi/2, np.pi/2)
        self.gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length", 0, 0.085, 0.04)
        self._bodies_idx = {}
        self.object_size = 0.06

        goal_range = 0.3
        obj_range = 0.05
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])  # goal_xy_range = 0.3,
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, 0])
        self.obj_range_low = np.array([-obj_range / 2, -obj_range / 2, 0])  # obj_xy_range = 0.05, originally 0.3  easy 0.05
        self.obj_range_high = np.array([obj_range / 2, obj_range / 2, 0])
        self.action_space_low = -1.0
        self.action_space_high = 1.0
        self.action_shape = 7
        self.task_input_shape = 6  # 3+3 (obj_position)+goal_pos
        self.goal_pos_shape = 3
        self.distance_threshold = 0.05
        self.reward_type = "sparse"
        self._max_episode_steps = 50

        self._create_scene()

    def set_lateral_friction(self, body: str, link: int, lateral_friction: float) -> None:
        """Set the lateral friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            lateral_friction (float): Lateral friction.
        """
        # self.physicsClient.changeDynamics(
        #     bodyUniqueId=self._bodies_idx[body],
        #     linkIndex=link,
        #     lateralFriction=lateral_friction,
        # )
        p.changeDynamics(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            lateralFriction=lateral_friction,
        )

    def set_spinning_friction(self, body: str, link: int, spinning_friction: float) -> None:
        """Set the spinning friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            spinning_friction (float): Spinning friction.
        """
        # self.physicsClient.changeDynamics(
        #     bodyUniqueId=self._bodies_idx[body],
        #     linkIndex=link,
        #     spinningFriction=spinning_friction,
        # )
        p.changeDynamics(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            spinningFriction=spinning_friction,
        )

    def _create_geometry(
        self,
        body_name: str,
        geom_type: int,
        mass: float = 0.0,
        position: Optional[np.ndarray] = None,
        ghost: bool = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
        visual_kwargs: Dict[str, Any] = {},
        collision_kwargs: Dict[str, Any] = {},
    ) -> None:
        """Create a geometry.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            geom_type (int): The geometry type. See self.physics_client.GEOM_<shape>.
            mass (float, optional): The mass in kg. Defaults to 0.
            position (np.ndarray, optional): The position, as (x, y, z). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
            visual_kwargs (dict, optional): Visual kwargs. Defaults to {}.
            collision_kwargs (dict, optional): Collision kwargs. Defaults to {}.
        """
        position = position if position is not None else np.zeros(3)
        # baseVisualShapeIndex = self.physicsClient.createVisualShape(geom_type, **visual_kwargs)
        baseVisualShapeIndex = p.createVisualShape(geom_type, **visual_kwargs)
        if not ghost:
            # baseCollisionShapeIndex = self.physicsClient.createCollisionShape(geom_type, **collision_kwargs)
            baseCollisionShapeIndex = p.createCollisionShape(geom_type, **collision_kwargs)
        else:
            baseCollisionShapeIndex = -1
        # self._bodies_idx[body_name] = self.physicsClient.createMultiBody(
        self._bodies_idx[body_name] = p.createMultiBody(
            baseVisualShapeIndex=baseVisualShapeIndex,
            baseCollisionShapeIndex=baseCollisionShapeIndex,
            baseMass=mass,
            basePosition=position,
        )

        if lateral_friction is not None:
            self.set_lateral_friction(body=body_name, link=-1, lateral_friction=lateral_friction)
        if spinning_friction is not None:
            self.set_spinning_friction(body=body_name, link=-1, spinning_friction=spinning_friction)

    def create_box(
        self,
        body_name: str,
        half_extents: np.ndarray,
        mass: float,
        position: np.ndarray,
        rgba_color: Optional[np.ndarray] = None,
        specular_color: Optional[np.ndarray] = None,
        ghost: bool = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
        texture: Optional[str] = None,
    ) -> None:
        """Create a box.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            half_extents (np.ndarray): Half size of the box in meters, as (x, y, z).
            mass (float): The mass in kg.
            position (np.ndarray): The position, as (x, y, z).
            rgba_color (np.ndarray, optional): Body color, as (r, g, b, a). Defaults as [0, 0, 0, 0]
            specular_color (np.ndarray, optional): Specular color, as (r, g, b). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
            texture (str or None, optional): Texture file name. Defaults to None.
        """
        rgba_color = rgba_color if rgba_color is not None else np.zeros(4)
        specular_color = specular_color if specular_color is not None else np.zeros(3)
        visual_kwargs = {
            "halfExtents": half_extents,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"halfExtents": half_extents}
        self._create_geometry(
            body_name,
            # geom_type=self.physicsClient.GEOM_BOX,
            geom_type=p.GEOM_BOX,
            mass=mass,
            position=position,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )
        if texture is not None:
            print("we don't have texture now.")

    def create_cylinder(
        self,
        body_name: str,
        radius: float,
        height: float,
        mass: float,
        position: np.ndarray,
        rgba_color: Optional[np.ndarray] = None,
        specular_color: Optional[np.ndarray] = None,
        ghost: bool = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
    ) -> None:
        """Create a cylinder.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            radius (float): The radius in meter.
            height (float): The height in meter.
            mass (float): The mass in kg.
            position (np.ndarray): The position, as (x, y, z).
            rgba_color (np.ndarray, optional): Body color, as (r, g, b, a). Defaults as [0, 0, 0, 0]
            specular_color (np.ndarray, optional): Specular color, as (r, g, b). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
        """
        rgba_color = rgba_color if rgba_color is not None else np.zeros(4)
        specular_color = specular_color if specular_color is not None else np.zeros(3)
        visual_kwargs = {
            "radius": radius,
            "length": height,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"radius": radius, "height": height}
        self._create_geometry(
            body_name,
            geom_type=p.GEOM_CYLINDER,
            mass=mass,
            position=position,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)

    def read_debug_parameter(self):
        # read the value of task parameter
        x = p.readUserDebugParameter(self.xin)
        y = p.readUserDebugParameter(self.yin)
        z = p.readUserDebugParameter(self.zin)
        roll = p.readUserDebugParameter(self.rollId)
        pitch = p.readUserDebugParameter(self.pitchId)
        yaw = p.readUserDebugParameter(self.yawId)
        gripper_opening_length = p.readUserDebugParameter(self.gripper_opening_length_control)

        return x, y, z, roll, pitch, yaw, gripper_opening_length

    def step(self, action, control_method='joint'):
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """
        assert control_method in ('joint', 'end')
        action = np.clip(action, self.action_space_low, self.action_space_high)
        # map the [-1, 1] normalized action to the real robot action space
        # which is low=[-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 0.0]
        # high=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1]
        # action = np.concatenate([action[:-1] * 0.2, [action[-1] * 0.05 + 0.05]])  #this is when we don't block the gripper
        # default bolck the gripper
        action = np.concatenate([action[:-1] * 0.05, [0.0]])

        self.robot.move_ee(action[:-1], control_method)
        self.robot.move_gripper(action[-1])  # limit the gripper action by 0 - 0.1
        for _ in range(20):  # Wait for a few steps when rendering, originally 120
            self.step_simulation()

        achieved_goal = self.get_achieved_goal()
        desired_goal = self.get_desired_goal()
        reward = self.compute_reward(achieved_goal, desired_goal, None)
        done = True if reward == 0 else False
        info = {"is_success": self.is_success(achieved_goal, desired_goal)}
        return self.get_obs(action[-1]), reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        # the info: Dict[str, Any] is useless but cannot be deleted
        # because when this function is called, they always pass a None to this info Dict
        d = self.distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float64)
        else:
            return -d

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        d = self.distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float64)

    def get_rgbd_obs(self):
        # this function get the observation from camera
        # includes rgb, depth, and segmentation image
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs.update(dict(rgb=rgb, depth=depth, seg=seg))
        else:
            assert self.camera is None
        obs.update(self.robot.get_joint_obs())

        return obs

    def get_obs(self, gripper_open_length):
        # position, rotation of the object
        object_position, object_rotation = p.getBasePositionAndOrientation(self._bodies_idx["object"])
        object_position = np.array(object_position)
        object_rotation = np.array(p.getEulerFromQuaternion(object_rotation))
        # observation = np.concatenate(
        #     [
        #         object_position,
        #         object_rotation,
        #     ]
        # )
        observation = object_position

        achieved_goal = self.get_achieved_goal()
        desired_goal = self.get_desired_goal()
        arm_joint_pos = np.array(self.robot.get_arm_joint_obs()["positions"])
        joint_pos = np.concatenate([arm_joint_pos, [gripper_open_length]])
        return {
            "observation": observation,
            "joint_pos": joint_pos,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }

    def get_achieved_goal(self) -> np.ndarray:
        object_position, _ = p.getBasePositionAndOrientation(self._bodies_idx["object"])
        object_position = np.array(object_position)
        return object_position

    def get_desired_goal(self) -> np.ndarray:
        object_position, _ = p.getBasePositionAndOrientation(self._bodies_idx["target"])
        object_position = np.array(object_position)
        return object_position

    def distance(self, a: np.ndarray, b: np.ndarray) -> Union[float, np.ndarray]:
        """Compute the distance between two array. This function is vectorized.

        Args:
            a (np.ndarray): First array.
            b (np.ndarray): Second array.

        Returns:
            Union[float, np.ndarray]: The distance between the arrays.
        """
        assert a.shape == b.shape
        return np.linalg.norm(a - b, axis=-1)

    def _create_scene(self):
        self.robot.reset()
        obj_orig = [0.0, 0.0, self.object_size / 2]
        obj_noise = np.random.uniform(self.obj_range_low, self.obj_range_high)
        goal_orig = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        goal_noise = np.random.uniform(self.goal_range_low, self.goal_range_high)

        self.create_cylinder(
            body_name="object",
            radius=0.0424264068712,
            height=0.06,
            mass=0.054,
            position=obj_orig + obj_noise,
            rgba_color=np.array([0.9, 0.1, 0.1, 1.0]),
            lateral_friction=0.72,  # 0.72
            spinning_friction=0.72,  # 0.72
        )

        self.create_cylinder(
            body_name="target",
            radius=0.0424264068712,
            height=0.06,
            mass=0.0,
            ghost=True,
            position=goal_orig + goal_noise,
            rgba_color=np.array([0.9, 0.1, 0.1, 0.3]),
        )

        self.create_cylinder(
            body_name="base",
            radius=0.08,
            height=0.0431,
            mass=0.0,
            ghost=True,
            position=np.array([-0.7, -0.1095, 0.0431 / 2]),
            rgba_color=np.array([0.75, 0.75, 0.75, 1.0]),
        )

    def reset(self):
        self.robot.reset()
        obj_orig = [0.0, 0.0, self.object_size / 2]
        obj_noise = np.random.uniform(self.obj_range_low, self.obj_range_high)
        goal_orig = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        goal_noise = np.random.uniform(self.goal_range_low, self.goal_range_high)
        p.resetBasePositionAndOrientation(self._bodies_idx["target"], goal_orig+goal_noise, np.array([0.0, 0.0, 0.0, 1.0]))
        p.resetBasePositionAndOrientation(self._bodies_idx["object"], obj_orig+obj_noise, np.array([0.0, 0.0, 0.0, 1.0]))
        return self.get_obs(self.robot.gripper_range[1])

    def close(self):
        p.disconnect(self.physicsClient)
