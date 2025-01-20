from collections import deque

import cv2
import numpy as np
import gymnasium as gym

from ocatari import OCAtari
from ocatari.ram.extract_ram_info import get_class_dict


class MaskedBaseWrapper(gym.ObservationWrapper):
    def __init__(self, env: OCAtari, buffer_window_size=4, *, include_pixels=False, num_planes=1):
        super().__init__(env)
        length = (num_planes + include_pixels) * buffer_window_size
        self.observation_space = gym.spaces.Box(0, 255.0, (length, 84, 84))

        self._buffer = deque([], maxlen=length)

        if include_pixels:
            self.init_obs = self.add_pixel_screen
        else:
            self.init_obs = lambda observation : observation

    def add_pixel_screen(self, observation):
        observation.append(self.env._ale.getScreenGrayscale())

    def observation(self, observation):
        self.init_obs(observation)
        for frame in observation:
            self._buffer.append(cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA))

        return np.array(self._buffer)

    def reset(self, *args, **kwargs):
        ret = super().reset(*args, **kwargs)
        for _ in range(self._buffer.maxlen):
            obs = self.observation(ret[0])

        return obs, *ret[1:]  # noqa: cannot be undefined



class BinaryMaskWrapper(MaskedBaseWrapper):
    def observation(self, observation):
        state = np.zeros((210, 160))
        for o in self.env.objects:  # noqa: type(env) == OCAtari
            if o is not None:
                x, y, w, h = o.xywh

                if x + w > 0 and y + h > 0:
                    for i in range(max(0, y), min(y + h, 209)):
                        for j in range(max(0, x), min(x + w, 159)):
                            state[i, j] = 255
        return super().observation([state])


class PixelMaskWrapper(MaskedBaseWrapper):
    def observation(self, observation):
        state = np.zeros((210, 160))
        gray_scale_img = self.env._ale.getScreenGrayscale()
        for o in self.env.objects:  # noqa: type(env) == OCAtari
            if o is not None:
                x,y,w,h = o.xywh

                if x+w > 0 and y+h > 0:
                    for i in range(max(0, y), min(y+h, 209)):
                        for j in range(max(0, x), min(x+w, 159)):
                            state[i, j] = gray_scale_img[i, j]
        return super().observation([state])


class ObjectTypeMaskWrapper(MaskedBaseWrapper):
    def __init__(self, env: OCAtari, *args, **kwargs):
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
        return super().observation([state])
