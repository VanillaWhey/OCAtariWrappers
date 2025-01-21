import numpy as np

from .masked_dqn import MaskedBaseWrapper


class DLWrapper(MaskedBaseWrapper):
    """
    Implements the proposed method by Davidson and Lake "Investigating Simple Object Representations in Model-Free Deep ReinforcementLearning" (2020).
    https://arxiv.org/abs/2002.06703
    """
    def __init__(self, env, buffer_window_size=4, *, include_pixels=False):
        super().__init__(env, buffer_window_size, include_pixels=include_pixels, num_planes=8)

    def observation(self, observation):
        img = self.unwrapped.ale.getScreenRGB()  # noqa: super test for ale
        dims = img.shape

        state = [np.ones((dims[0], dims[1])) * 255 for _ in range(8)]

        for o in self.env.objects:  # noqa: type(env) == OCAtari
            if not (o is None or o.category == "NoObject"):
                x, y, w, h = o.xywh
                if x + w > 0 and y + h > 0:
                    for i in range(max(0, y), min(y + h, 209)):
                        for j in range(max(0, x), min(x + w, 159)):
                            # Igloo
                            if o.category == "House" and list(img[i, j, :]) == [142, 142, 142]:
                                state[7][i, j] = 0
                            # Player
                            elif list(img[i, j, :]) in [[162, 98, 33], [162, 162, 42], [198, 108, 58], [142, 142, 142]]:
                                state[0][i, j] = 0
                            # Bad Animal
                            elif list(img[i, j, :]) in [[132, 144, 252], [210, 210, 64], [213, 130, 74]]:
                                state[1][i, j] = 0
                            # Land
                            elif list(img[i, j, :]) in [[192, 192, 192], [74, 74, 74]]:
                                state[2][i, j] = 0
                            # Bear
                            elif o.category == "Bear" and list(img[i, j, :]) in [[111, 111, 111], [214, 214, 214]]:
                                state[3][i, j] = 0
                            # Unvisited Floes
                            elif list(img[i, j, :]) == [214, 214, 214]:
                                state[4][i, j] = 0
                            # Visited Flows
                            elif list(img[i, j, :]) == [84, 138, 210]:
                                state[5][i, j] = 0
                            # Good Animal
                            elif list(img[i, j, :]) == [111, 210, 111]:
                                state[6][i, j] = 0

        return self.create_obs(state)