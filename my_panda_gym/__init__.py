import os

from gym.envs.registration import register

with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

for reward_type in ["sparse", "dense"]:
    for control_type in ["ee", "joints"]:
        reward_suffix = "Dense" if reward_type == "dense" else ""
        control_suffix = "Joints" if control_type == "joints" else ""
        kwargs = {"reward_type": reward_type, "control_type": control_type}

        register(
            id="PandaReach{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaL3Reach{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaL3ReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaL5Reach{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaL5ReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaReach{}{}-v3".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaReachEnv3",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaReach{}{}-v4".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaReachEnv4",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaL3Reach{}{}-v4".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaL3ReachEnv4",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaL5Reach{}{}-v4".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaL5ReachEnv4",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPush{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaPushEnv1",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPush{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaPushEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPush{}{}-v3".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaPushEnv3",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPush{}{}-v4".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaPushEnv4",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPushForward{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaPushForwardEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPushBackward{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaPushBackwardEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPushLeft{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaPushLeftEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPushRight{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaPushRightEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPushFrontStill{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaPushFrontStillEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPushBackStill{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaPushBackStillEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPushRightStill{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaPushRightStillEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPushLeftStill{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaPushLeftStillEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPushFrontRange{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaPushFrontRangeEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPushBackRange{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaPushBackRangeEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPushRightRange{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaPushRightRangeEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPushLeftRange{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaPushLeftRangeEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPushFricRange{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaPushFricRangeEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPushRocks{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaPushRocksEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPushMud{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaPushMudEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaSlide{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaSlideEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaSlide{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaSlideEnv1",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaSlide{}{}-v3".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaSlideEnv3",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPickAndPlace{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaPickAndPlaceEnv",
            kwargs=kwargs,
            max_episode_steps=50,  # originally 50
        )

        register(
            id="PandaPickAndPlace{}{}-v3".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaPickAndPlaceEnv3",
            kwargs=kwargs,
            max_episode_steps=50,  # originally 50
        )

        register(
            id="PandaPickAndPlace{}{}-v4".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaPickAndPlaceEnv4",
            kwargs=kwargs,
            max_episode_steps=50,  # originally 50
        )

        register(
            id="PandaStack{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaStackEnv",
            kwargs=kwargs,
            max_episode_steps=100,
        )

        register(
            id="PandaFlip{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaFlipEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaL5Push{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaL5PushEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaL5Push{}{}-v3".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaL5PushEnv3",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaL5Push{}{}-v4".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaL5PushEnv4",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaL5PushRocks{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaL5PushRocksEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaL5PickAndPlace{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaL5PickAndPlaceEnv",
            kwargs=kwargs,
            max_episode_steps=50,  # originally 50
        )

        register(
            id="PandaL5PickAndPlace{}{}-v3".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaL5PickAndPlaceEnv3",
            kwargs=kwargs,
            max_episode_steps=50,  # originally 50
        )

        register(
            id="PandaL3Push{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaL3PushEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaL3Push{}{}-v3".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaL3PushEnv3",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaL3Push{}{}-v4".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaL3PushEnv4",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaL3PushRocks{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaL3PushRocksEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaL3PickAndPlace{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaL3PickAndPlaceEnv",
            kwargs=kwargs,
            max_episode_steps=50,  # originally 50
        )

        register(
            id="PandaL3PickAndPlace{}{}-v3".format(control_suffix, reward_suffix),
            entry_point="my_panda_gym.envs:PandaL3PickAndPlaceEnv3",
            kwargs=kwargs,
            max_episode_steps=50,  # originally 50
        )
