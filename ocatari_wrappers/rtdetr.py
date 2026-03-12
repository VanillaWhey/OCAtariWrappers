from ultralytics import RTDETR

import gymnasium as gym
from .yolo import CategoryGameObject

class RTDETROCAtariWrapper(gym.ObservationWrapper):
    """
    An object extraction wrapper using RT-DETR to imitate OCAtari for downstream
    masking. Needs ori (210x160) input images.
    """
    def __init__(self, env, model_path):
        super().__init__(env)
        self.model = RTDETR(model_path)
        self.objects = []

    def observation(self, observation):
        self.objects = []
        result = self.model(observation)[0]
        for box in result.boxes:
            center_wh = box.xywh.cpu().detach()
            xywh = center_wh[0]
            xywh[:2] = xywh[:2] - xywh[2:] / 2  # left upper corner
            xywh = xywh.int().numpy()
            self.objects.append(CategoryGameObject(result.names[box.cls.int().item()], xywh))
        return observation
