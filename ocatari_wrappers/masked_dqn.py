import cv2
import numpy as np
import gymnasium as gym
from collections import deque

from ocatari.ram.extract_ram_info import get_class_dict, get_max_objects


class MaskedBaseWrapper(gym.ObservationWrapper):
    """
    Base class for all our wrappers.
    """

    def __init__(self, env, buffer_window_size=4, *, include_pixels=False, num_planes=1, work_in_output_shape=False, needs_pixels=False):
        """
        Args:
            env (gym.Env, OCAtari): The environment to wrap (Should contain an OCAtari in the stack).
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

    def __init__(self, env: gym.Env, *args, v2=False, **kwargs):
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

    def __init__(self, env: gym.Env, *args, v2=False, **kwargs):
        if v2:
            self.object_types = {k: i for i, k in enumerate(get_max_objects(env.game_name, env.hud).keys())}  # noqa: OCAtari in the env stack
        else:
            self.object_types = {k: i for i, k in enumerate(get_class_dict(env.game_name).keys())}  # noqa: OCAtari in the env stack
        super().__init__(env, num_planes=len(self.object_types), *args, **kwargs)

    def set_value(self, y_min, y_max, x_min, x_max, o):
        self.state[self.object_types[o.category], y_min:y_max, x_min:x_max].fill(255)


class PixelMaskPlanesWrapper(MaskedBaseWrapper):
    """
    A Wrapper that removes the background and only includes the bounding
    boxes of all objects filled with their grayscale pixels, where
    every object type is on its own plane.
    """

    def __init__(self, env: gym.Env, *args, **kwargs):
        self.object_types = {k: i for i, k in enumerate(get_max_objects(env.game_name, env.hud).keys())}  # noqa: OCAtari in the env stack
        super().__init__(env, num_planes=len(self.object_types), *args, work_in_output_shape=False, needs_pixels=True, **kwargs)

    def set_value(self, y_min, y_max, x_min, x_max, o):
        self.state[self.object_types[o.category], y_min:y_max, x_min:x_max] = self.pixel_screen[y_min:y_max, x_min:x_max]
