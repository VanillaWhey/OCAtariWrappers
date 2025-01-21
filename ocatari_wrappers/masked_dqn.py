from collections import deque

import cv2
import numpy as np
import gymnasium as gym

from ocatari.ram.extract_ram_info import get_class_dict


class MaskedBaseWrapper(gym.ObservationWrapper):
    """
    Base class for all our wrappers.
    """

    def __init__(self, env, buffer_window_size=4, *, include_pixels=False, num_planes=1):
        """
        Args:
            env (gym.Env, OCAtari): The environment to wrap (Should contain an OCAtari in the stack).
            buffer_window_size (int): How many observations to stack.
            include_pixels (bool): If True, a grayscale screen is added to the observations.
            num_planes (int): The number of planes that this wrapper will produce (only important for subclasses).
        """
        super().__init__(env)
        try:
            env.unwrapped.ale  # noqa: test for ale
            env.objects  # noqa: test for objects
        except AttributeError as e:
            raise AttributeError("Please use OCAtari with this wrapper.") from e

        length = (num_planes + include_pixels) * buffer_window_size
        self.observation_space = gym.spaces.Box(0, 255.0, (length, 84, 84))

        self._buffer = deque([], maxlen=length)

        if include_pixels:
            self.init_obs = self.add_pixel_screen
        else:
            self.init_obs = lambda observation: observation

    def add_pixel_screen(self, observation):
        """
        Adds a grayscale image of the game screen to the observations.

        Args:
            observation (np.ndarray): The observations to extend.

        Returns:
            np.ndarray: The original observation with a game screen plane added.
        """
        observation.append(self.unwrapped.ale.getScreenGrayscale())  # noqa: OCAtari in the stack

    def create_obs(self, obs_plane_list):
        """
        Resizes all the feature planes from the subclasses.

        Args:
            obs_plane_list (list): A list of all masked planes.

        Returns:
            np.ndarray: The final observations of shape Yx84x84.
        """
        
        # add grayscale screen if necessary
        self.init_obs(obs_plane_list)
        
        # resize all observation planes
        for frame in obs_plane_list:
            self._buffer.append(cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA))

        return np.array(self._buffer)

    def reset(self, *args, **kwargs):
        ret = super().reset(*args, **kwargs)
        
        # fill buffer
        for _ in range(self._buffer.maxlen):
            obs = self.observation(ret[0])

        return obs, *ret[1:]  # noqa: cannot be undefined



class BinaryMaskWrapper(MaskedBaseWrapper):
    """
    A Wrapper that outputs a binary mask including
    only white bounding boxes of all objects on a black background.
    """
    
    def observation(self, observation):
        state = np.zeros((210, 160))
        for o in self.env.objects:  # noqa: OCAtari in the stack
            if o is not None:
                x, y, w, h = o.xywh

                if x + w > 0 and y + h > 0:
                    for i in range(max(0, y), min(y + h, 209)):
                        for j in range(max(0, x), min(x + w, 159)):
                            state[i, j] = 255
        return self.create_obs([state])


class PixelMaskWrapper(MaskedBaseWrapper):
    """
    A Wrapper that removes the background and only includes the bounding
    boxes of all objects filled with their grayscale pixels.
    """

    def observation(self, observation):
        state = np.zeros((210, 160))
        gray_scale_img = self.unwrapped.ale.getScreenGrayscale()  # noqa: OCAtari in the stack
        for o in self.env.objects:  # noqa: type(env) == OCAtari
            if o is not None:
                x,y,w,h = o.xywh

                if x+w > 0 and y+h > 0:
                    for i in range(max(0, y), min(y+h, 209)):
                        for j in range(max(0, x), min(x+w, 159)):
                            state[i, j] = gray_scale_img[i, j]
        return self.create_obs([state])


class ObjectTypeMaskWrapper(MaskedBaseWrapper):
    """
    A Wrapper that outputs a grayscale mask including
    only filled bounding boxes of all objects on a black background where
    each object type has a different shade of gray.
    """

    def __init__(self, env: gym.Env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.object_types =  list(dict.fromkeys(get_class_dict(self.game_name)))


    def observation(self, observation):
        state = np.zeros((210, 160))
        for o in self.env.objects:  # noqa: type(env) == OCAtari
            if not (o is None or o.category == "NoObject"):
                x, y, w, h = o.xywh
                value = 255 * (1 + self.object_types.index(o.category)) // len(self.object_types)

                if x + w > 0 and y + h > 0:
                    for i in range(max(0, y), min(y + h, 209)):
                        for j in range(max(0, x), min(x + w, 159)):
                            state[i, j] = value
        return self.create_obs([state])
