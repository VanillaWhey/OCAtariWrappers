import cv2
import numpy as np
import gymnasium as gym
from collections import deque

from gymnasium.core import ObsType, WrapperObsType
from ocatari.ram.extract_ram_info import get_class_dict, get_max_objects
from ocatari.ram import GameObject


class MaskedBaseWrapper(gym.ObservationWrapper):
    """
    Base class for all our wrappers.
    """

    def __init__(self, env, buffer_window_size=4, *, include_pixels=False, num_planes=1, work_in_output_shape=False, needs_pixels=False):
        """
        Args:
            env (gym.Env, OCAtari): The environment to wrap. (OCAtari needs to be in the stack)
            buffer_window_size (int): How many observations to stack.
            include_pixels (bool): If True, a grayscale screen is added to the observations.
            num_planes (int): The number of planes that this wrapper will produce (only important for subclasses).
            work_in_output_shape (bool): Directly work in the 84x84 planes instead of downscaling the produced 210x160 ones.
            needs_pixels (bool): If True, the new grayscale game screen is collected every time.
        """
        super().__init__(env)
        try:
            env.unwrapped.ale  # noqa: test for ale
            env.objects  # noqa: test for objects
        except AttributeError as e:
            raise AttributeError("Please use OCAtari with this wrapper.") from e

        length = (num_planes + include_pixels) * buffer_window_size
        self.observation_space = gym.spaces.Box(0, 255.0, (length, 84, 84))
        self.buffer_window_size = buffer_window_size
        self._buffer = deque([], maxlen=length)

        # for saving the current grayscale game screen and masked state
        self.pixel_screen = None
        self.state = None

        if work_in_output_shape:  # directly create the 84x84 frames
            self.working_shape = (num_planes + include_pixels, 84, 84)
            self.calc_limits = lambda x, y, x_w, y_h: (
                max(0, y * 84 // 210),
                min(y_h * 84 // 210 + 1, 84),
                max(0, x * 84 // 160),
                min(x_w * 84 // 160 + 1, 84)
            )
            self.maybe_scale = lambda l: l  # no downscaling necessary
            if include_pixels:
                self.maybe_add_pixel_screen = self.add_pixel_screen_new  # add downscaled grayscale game screen
                needs_pixels = True
            else:
                self.maybe_add_pixel_screen = lambda: None
        else:  # create 210x160 frames and then downscale them
            self.working_shape = (num_planes + include_pixels, 210, 160)
            self.calc_limits = lambda x, y, x_w, y_h: (
                max(0, y),
                min(y_h, 210),
                max(0, x),
                min(x_w, 160)
            )
            self.maybe_scale = lambda l: [cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA) for frame in l]  # downscale frames
            if include_pixels:
                self.maybe_add_pixel_screen = self.add_pixel_screen_org  # add original grayscale game screen
                needs_pixels = True
            else:
                self.maybe_add_pixel_screen = lambda: None

        if needs_pixels:
            self.current_pixel_screen = lambda: self.unwrapped.ale.getScreenGrayscale()  # noqa: OCAtari in the env stack
        else:
            self.current_pixel_screen = lambda: None


    def observation(self, observation):
        self.state = np.zeros(self.working_shape, dtype=np.uint8)
        self.pixel_screen = self.current_pixel_screen()  # noqa: only used when somethis is returned
        for o in self.env.objects:  # noqa: OCAtari in the stack
            if not (o is None or o.category == "NoObject"):
                x, y, w, h = o.xywh
                x_w = x + w
                y_h = y + h
                if x_w > 0 and y_h > 0:
                    self.set_value(*self.calc_limits(x, y, x_w, y_h), o)
        return self.create_obs(self.state)

    def add_pixel_screen_org(self):
        """
        Adds a grayscale image of the game screen to the observations.
        """
        self.state[-1] = self.pixel_screen

    def add_pixel_screen_new(self):
        """
        Adds a downscaled grayscale image of the game screen to the observations.
        """
        self.state[-1] = cv2.resize(
            self.pixel_screen,
            (84, 84),
            interpolation=cv2.INTER_AREA
        )

    def create_obs(self, obs_planes):
        """
        Creates the final observations, i.e.,
        adding the grayscale image if wanted and doing the frame stacking.

        Args:
            obs_planes (np.ndarray): The masked planes.

        Returns:
            np.ndarray: The final observations of shape Yx84x84.
        """
        self.maybe_add_pixel_screen()
        self._buffer.extend(self.maybe_scale(obs_planes))
        return np.asarray(self._buffer)

    def reset(self, *args, **kwargs):
        ret = super().reset(*args, **kwargs)

        # fill buffer
        for _ in range(self.buffer_window_size):
            obs = self.observation(ret[0])

        return obs, *ret[1:]  # noqa: cannot be undefined

    def set_value(self, y_min, y_max, x_min, x_max, o):
        raise NotImplementedError


class BinaryMaskWrapper(MaskedBaseWrapper):
    """
    A Wrapper that outputs a binary mask including
    only white bounding boxes of all objects on a black background.
    """

    def set_value(self, y_min, y_max, x_min, x_max, o):
        self.state[0, y_min:y_max, x_min:x_max].fill(255)


class PixelMaskWrapper(MaskedBaseWrapper):
    """
    A Wrapper that removes the background and only includes the bounding
    boxes of all objects filled with their grayscale pixels.
    """
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, work_in_output_shape=False, needs_pixels=True, **kwargs)

    def set_value(self, y_min, y_max, x_min, x_max, o):
        self.state[0, y_min:y_max, x_min:x_max] = self.pixel_screen[y_min:y_max, x_min:x_max]


class ObjectTypeMaskWrapper(MaskedBaseWrapper):
    """
    A Wrapper that outputs a grayscale mask including
    only filled bounding boxes of all objects on a black background where
    each object type has a different shade of gray.
    """
    def __init__(self, env, *args, v2=False, **kwargs):
        """
        :param v2: Only use HUD objects if HUD is specified.
        """
        super().__init__(env, *args, **kwargs)
        if v2:
            keys = get_max_objects(env.game_name, env.hud).keys()  # noqa: OCAtari in the env stack
        else:
            keys = get_class_dict(env.game_name).keys()  # noqa: OCAtari in the env stack
        shades = 255 // len(keys)
        self.object_types =  {k: (i + 1) * shades for i, k in enumerate(keys)}

    def set_value(self, y_min, y_max, x_min, x_max, o):
        self.state[0, y_min:y_max, x_min:x_max].fill(self.object_types[o.category])


class ObjectTypeMaskPlanesWrapper(MaskedBaseWrapper):
    """
    A Wrapper that outputs a binary mask including
    only white bounding boxes of all objects on a black background, where
    every object type is on its own plane.
    """
    def __init__(self, env, *args, v2=False, extra_planes=0, **kwargs):
        """
        :param v2: Only use HUD objects if HUD is specified.
        """
        if v2:
            self.object_types = {k: i for i, k in enumerate(get_max_objects(env.game_name, env.hud).keys())}  # noqa: OCAtari in the env stack
        else:
            self.object_types = {k: i for i, k in enumerate(get_class_dict(env.game_name).keys())}  # noqa: OCAtari in the env stack
        super().__init__(env, num_planes=len(self.object_types) + extra_planes, *args, **kwargs)

    def set_value(self, y_min, y_max, x_min, x_max, o):
        self.state[self.object_types[o.category], y_min:y_max, x_min:x_max].fill(255)

class BigPlaneWrapper(MaskedBaseWrapper):
    """
    A Wrapper that outputs a binary mask including
    only white bounding boxes of all objects on a black background, where
    every object type is on its own 84x84 area in the big plane (84x(84n),
    n = number of object types).
    """
    def __init__(self, env, *args, **kwargs):
        self.object_types = {k: i for i, k in enumerate(get_max_objects(env.game_name, env.hud).keys())}  # noqa: OCAtari in the env stack
        super().__init__(env, work_in_output_shape=True, *args, **kwargs)
        length = len(self.object_types) * 84
        self.working_shape = (self.buffer_window_size, 84, length)
        self.observation_space = gym.spaces.Box(0, 255.0, self.working_shape)

    def set_value(self, y_min, y_max, x_min, x_max, o):
        offset = self.object_types[o.category] * 84
        self.state[0, y_min:y_max, offset + x_min: offset + x_max].fill(255)


class PixelMaskPlanesWrapper(MaskedBaseWrapper):
    """
    A Wrapper that removes the background and only includes the bounding
    boxes of all objects filled with their grayscale pixels, where
    every object type is on its own plane.
    """
    def __init__(self, env, *args, **kwargs):
        self.object_types = {k: i for i, k in enumerate(get_max_objects(env.game_name, env.hud).keys())}  # noqa: OCAtari in the env stack
        super().__init__(env, num_planes=len(self.object_types), *args, work_in_output_shape=False, needs_pixels=True, **kwargs)

    def set_value(self, y_min, y_max, x_min, x_max, o):
        self.state[self.object_types[o.category], y_min:y_max, x_min:x_max] = self.pixel_screen[y_min:y_max, x_min:x_max]


class ImperfectDetectionWrapper(gym.ObservationWrapper):
    """
    A wrapper to simulate an imperfect object detector.
    """
    class MislabeledGameObject(GameObject):
        """
        A GameObject that can be of any category.
        """
        def __init__(self, category, xywh):
            super().__init__()
            self.xywh = xywh
            self._category = category

        @property
        def category(self):
            return self._category


    def __init__(self, env, mislabeling_probability=0.1, failed_detection_probability=0.1, noise_std=1.0):
        """
        :param env (gym.Env, OCAtari): The environment to wrap. (OCAtari needs to be in the stack)
        :param mislabeling_probability (float): The probability of mislabeling the object into another random category.
        :param failed_detection_probability (float): The probability of not detecting an object.
        :param noise_std (float): The standard deviation for the Gaussian noise added to the xywh values.
        """
        super().__init__(env)
        self.mislabeling_probability = mislabeling_probability
        self.failed_detection_probability = failed_detection_probability
        self.noise_std = noise_std
        self.objects = []
        self.categories = list(get_max_objects(env.game_name, env.hud).keys())  # noqa: OCAtari in the env stack


    def observation(self, observation):
        self.objects = []
        for o in self.env.objects:  # noqa: OCAtari in the stack
            if (not (o is None or o.category == "NoObject")
                    and self.np_random.random() > self.failed_detection_probability): # failed detection
                # noisy detection
                xywh = o.xywh + self.np_random.normal(scale=self.noise_std, size=4).astype(int)
                xywh = np.maximum([-xywh[2] + 1, -xywh[3] + 1, 1, 1], xywh)
                # mislabelled object
                if self.np_random.random() <= self.mislabeling_probability:
                    c = self.np_random.choice(self.categories)
                else:
                    c = o.category
                o = ImperfectDetectionWrapper.MislabeledGameObject(c, xywh)

                self.objects.append(o)
        return observation


class MultiOCCAMWrapper(gym.ObservationWrapper):
    def __init__(self, env, wrappers, **kwargs):

        class DQNWrapper(gym.ObservationWrapper):
            def __init__(self, env):
                super().__init__(env)
                self._buffer = deque(maxlen=4)
                self.observation_space = gym.spaces.Box(0, 255, shape=(4, 84, 84))

            def observation(self, observation):
                state = self.unwrapped.ale.getScreenGrayscale()  # noqa: there must be an ALE
                state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
                self._buffer.append(state)
                return np.asarray(self._buffer)

            def reset(self, *args, **kwargs):
                ret = super().reset(*args, **kwargs)

                # fill buffer
                for _ in range(4):
                    obs = self.observation(ret[0])

                return obs, *ret[1:]  # noqa: cannot be undefined


        class OCAtariOBJWrapper(gym.Wrapper):
            def __init__(self, env):
                super().__init__(env)
                assert env.create_ns_stack  # noqa: must be wrapper around OCAtari anyway
                self._buffer = env._state_buffer_ns  # noqa: must be wrapper around OCAtari anyway


        mapping = {
            "plane_masks": ObjectTypeMaskPlanesWrapper,
            "class_masks": ObjectTypeMaskWrapper,
            "binary_masks": BinaryMaskWrapper,
            "object_masks": PixelMaskWrapper,
            "pixel_planes": PixelMaskPlanesWrapper,
            "big_planes": BigPlaneWrapper,
            "dqn": DQNWrapper,
            "obj": OCAtariOBJWrapper,
        }

        self.wrappers = {}
        spaces = {}
        for key in wrappers:
            env = mapping[key](env, **kwargs)
            self.wrappers[key] = env
            spaces[key] = env.observation_space

        super().__init__(env)
        self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, observation):
        return {key: np.asarray(self.wrappers[key]._buffer) for key in self.wrappers}  # noqa: private property of class in this file


class MultiplyInputWrapper(gym.ObservationWrapper):
    def __init__(self, env, num_channels):
        super().__init__(env)
        self.num_channels = num_channels
        c, w, h = self.env.observation_space.shape
        self._observation_space = gym.spaces.Box(0, 255, (num_channels * c, w, h))

    def observation(self, observation: ObsType):
        return np.concatenate([observation] * self.num_channels)