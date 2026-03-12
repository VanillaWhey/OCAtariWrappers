from ocatari.ram import GameObject
from ultralytics import YOLO

import gymnasium as gym
import numpy as np
import cv2


class CategoryGameObject(GameObject):
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

class YOLOCAtariWrapper(gym.ObservationWrapper):
    """
    An object extraction wrapper using YOLO to imitate OCAtari for downstream
    masking. Needs ori (210x160) input images.
    """
    def __init__(self, env, model_path, upscale=False):
        super().__init__(env)
        self.model = YOLO(model_path)
        self.objects = []
        self.upscale = upscale

    def observation(self, observation):
        self.objects = []
        if self.upscale:
            new_img = np.zeros((630, 630, 3), dtype=np.uint8)
            new_img[:, :480, :] = cv2.resize(observation, (480, 630), interpolation=cv2.INTER_LINEAR)
            observation = new_img
        result = self.model(observation[...,::-1])[0]  # channel order is different from training
        for box in result.boxes:
            center_wh = box.xywh.cpu().detach()
            xywh = center_wh[0]
            xywh[:2] = xywh[:2] - xywh[2:] / 2  # left upper corner
            if self.upscale:
                xywh /= 3
            xywh = xywh.int().numpy()
            self.objects.append(CategoryGameObject(result.names[box.cls.int().item()], xywh))
        return observation
