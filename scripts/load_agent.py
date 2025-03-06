from argparse import ArgumentParser
from functools import partial
from gzip import GzipFile
from pathlib import Path
import numpy as np
import random
import pygame
from functools import partial

try:
    import torch
    from torch import nn
    from torch.distributions.categorical import Categorical
    torch_imported = True
except ModuleNotFoundError:
    torch_imported = False


test_parser = ArgumentParser()
test_parser.add_argument("-p", "--path", type=str, default=None,
                         help="path to the model")
test_parser.add_argument("-g", "--game", type=str, required=True,
                         help="game to evaluate (e.g. 'Pong')")
test_parser.add_argument("-i", "--iou", type=float, default=0.8,
                         help="Minimum iou for image saving (e.g. 0.7)")
test_parser.add_argument("-s", "--seed", type=float, default=None,
                         help="If provided, set the seed")


ROT_MATRIX = np.array([[0, -1], [1, 0]])


def make_deterministic(seed, mdp, states_dict=None):
    random.seed(seed)
    np.random.seed(seed)
    mdp.seed(seed)
    if torch_imported:
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if torch_imported:
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    class PPOAgent(nn.Module):
        def __init__(self, env):
            super().__init__()
            self.network = nn.Sequential(
                layer_init(
                    nn.Conv2d(env.observation_space.shape[0], 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, 512)),
                nn.ReLU(),
            )
            self.actor = layer_init(
                nn.Linear(512, env.action_space.n), std=0.01)
            self.critic = layer_init(nn.Linear(512, 1), std=1)

        def get_value(self, x):
            return self.critic(self.network(x / 255.0))

        def get_action_and_value(self, x, action=None):
            hidden = self.network(x / 255.0)
            logits = self.actor(hidden)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

        def draw_action(self, state):
            return self.get_action_and_value(state)[0]

    class PPObj(nn.Module):
        def __init__(self, envs, device, encoder_dims=(256, 512, 1024, 1024, 512, 256), decoder_dims=(256,)):
            super().__init__()
            self.device = device

            dims = envs.observation_space.shape
            layers = nn.ModuleList()

            in_dim = dims[-1]

            for l in encoder_dims:
                layers.append(layer_init(nn.Linear(in_dim, l)))
                layers.append(nn.ReLU())
                in_dim = l
            layers.append(nn.Flatten())
            in_dim *= np.prod(dims[:-1], dtype=int)
            l = in_dim
            for l in decoder_dims:
                layers.append(layer_init(nn.Linear(in_dim, l)))
                layers.append(nn.ReLU())
                in_dim = l

            self.network = nn.Sequential(*layers)
            self.actor = layer_init(
                nn.Linear(l, envs.action_space.n), std=0.01)
            self.critic = layer_init(nn.Linear(l, 1), std=1)

        def get_value(self, x):
            return self.critic(self.network(x))

        def get_action_and_value(self, x, action=None):
            hidden = self.network(x)
            logits = self.actor(hidden)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

        def draw_action(self, x, states=None, **_):
            return self.get_action_and_value(x)[0]

    class RandomAgent():
        """
        A agent acting randomly (following a uniform distribution).

        :param nb_actions
        """

        def __init__(self, nb_actions) -> None:
            self.nb_actions = nb_actions

        def draw_action(self, *args, **kwargs) -> int:
            return random.randint(0, self.nb_actions-1)


def _load_checkpoint(fpath, device="cpu"):
    fpath = Path(fpath)
    with fpath.open("rb") as file:
        with GzipFile(fileobj=file) as inflated:
            return torch.load(inflated, map_location=device)


def _epsilon_greedy(obs, model, eps=0.001):
    if torch.rand((1,)).item() < eps:
        return torch.randint(model.action_no, (1,)).item(), None
    obs = obs.byte()
    q_val, argmax_a = model(obs).max(1)
    return argmax_a.item(), q_val


def load_agent(opt, env=None, device="cpu"):
    pth = opt if isinstance(opt, str) else opt.path
    if device == "cpu":
        ckpt = torch.load(pth, map_location=torch.device('cpu'))
    else:
        ckpt = torch.load(pth)
    if "ppo" in pth and env.obs_mode == "dqn":
        agent = PPOAgent(env)
        agent.load_state_dict(ckpt["model_weights"])
    elif env.obs_mode == "obj":
        agent = PPObj(env, device)
        agent.load_state_dict(ckpt["model_weights"])
    else:
        return None

    policy = agent.draw_action

    return agent, policy


def draw_arrow(surface: pygame.Surface, start_pos: (float, float), end_pos: (float, float),
               tip_length: int = 6, tip_width: int = 6, **kwargs):
    start_pos = np.asarray(start_pos)
    end_pos = np.asarray(end_pos)

    # Arrow body
    pygame.draw.line(surface, start_pos=start_pos, end_pos=end_pos, **kwargs)

    # Arrow tip
    arrow_dir = end_pos - start_pos
    arrow_dir_norm = arrow_dir / np.linalg.norm(arrow_dir)
    tip_anchor = end_pos - tip_length * arrow_dir_norm

    left_tip_end = tip_anchor + tip_width / 2 * \
        np.matmul(ROT_MATRIX, arrow_dir_norm)
    right_tip_end = tip_anchor - tip_width / \
        2 * np.matmul(ROT_MATRIX, arrow_dir_norm)

    pygame.draw.line(surface, start_pos=left_tip_end,
                     end_pos=end_pos, **kwargs)
    pygame.draw.line(surface, start_pos=right_tip_end,
                     end_pos=end_pos, **kwargs)


def draw_label(surface: pygame.Surface, text: str, position: (int, int), font: pygame.font.SysFont):
    """Renders a framed label text to a pygame surface."""
    text = font.render(text, True, (255, 255, 255), None)
    text_rect = text.get_rect()

    frame_rect = text_rect.copy()
    frame_rect.topleft = position
    frame_rect.w += 5
    frame_rect.h += 6

    frame_surface = pygame.Surface((frame_rect.w, frame_rect.h))
    frame_surface.set_alpha(80)  # make transparent

    # Draw label background
    frame_surface.fill((0, 0, 0))
    surface.blit(frame_surface, position)

    # Draw text
    text_rect.topleft = position[0] + 3, position[1] + 3
    surface.blit(text, text_rect)


def draw_orientation_indicator(surface: pygame.Surface, orientation_value: int,
                               x_c: int, y_c: int, w: int, h: int):
    center = np.asarray([x_c, y_c])
    alpha = orientation_value / 8 * np.pi  # orientation angle (in radians)

    triangle = np.array([[0, -18], [-6, -7], [6, -7]])
    rot = get_rotation_matrix(alpha)
    triangle = center + np.dot(rot, triangle.T).T  # transform

    pygame.draw.circle(surface, (255, 30, 180), (x_c, y_c), 10)
    pygame.draw.polygon(surface, (255, 30, 180), triangle)


def get_rotation_matrix(rad: float):
    return np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
