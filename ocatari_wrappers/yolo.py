from ocatari.ram import GameObject
from ultralytics import YOLO

import gymnasium as gym


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
    def __init__(self, env, model_path):
        super().__init__(env)
        self.model = YOLO(model_path)
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
