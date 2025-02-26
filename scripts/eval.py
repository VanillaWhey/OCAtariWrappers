from hackatari import HackAtari
import numpy as np
import torch
import gymnasium as gym
from load_agent import load_agent
import os
import argparse
import hackatari
from ocatari_wrappers import BinaryMaskWrapper, PixelMaskWrapper, ObjectTypeMaskWrapper, ObjectTypeMaskPlanesWrapper, PixelMaskPlanesWrapper
from stable_baselines3.common.atari_wrappers import (
    FireResetEnv,
)
import sys

# Disable graphics window (SDL) for headless execution
os.environ["SDL_VIDEODRIVER"] = "dummy"


class HackAtariArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        # Check if `-h` or `--help` is in the arguments
        if args is None:
            args = sys.argv[1:]
        if '-h' in args or '--help' in args:
            if not '-g' in args or '--game' in args:
                print(
                    "Call the script with a given game to get a list of available modifications.")
            else:
                print(hackatari._available_modifications(
                    args[args.index('-g') + 1]))
                print(
                    "\n provide -h (or --help) without a game argument for the original help message.")
                exit(0)

        # Call the original `parse_args` method to display the default help
        return super().parse_args(args, namespace)


def combine_means_and_stds(mu_list, sigma_list, n_list):
    """
    Combine multiple means and standard deviations using their respective sample sizes.

    Args:
        mu_list (list): List of means.
        sigma_list (list): List of standard deviations.
        n_list (list): List of sample sizes.

    Returns:
        tuple: Combined mean and combined standard deviation.
    """
    if not (len(mu_list) == len(sigma_list) == len(n_list)):
        raise ValueError("All input lists must have the same length.")

    total_n = sum(n_list)
    combined_mean = sum(n * mu for mu, n in zip(mu_list, n_list)) / total_n
    combined_variance = sum(
        n * (sigma**2 + (mu - combined_mean)**2)
        for mu, sigma, n in zip(mu_list, sigma_list, n_list)
    ) / total_n
    combined_std = np.sqrt(combined_variance)

    return combined_mean, combined_std


def main():
    """Main function to run HackAtari experiments with different agents."""
    parser = HackAtariArgumentParser(description="HackAtari Experiment Runner")

    # Game and environment parameters
    parser.add_argument("-g", "--game", type=str,
                        default="Seaquest", help="Game to be run")
    parser.add_argument("-obs", "--obs_mode", type=str,
                        default="dqn", help="Observation mode (ori, dqn, obj)")
    parser.add_argument("-w", "--window", type=int, default=4,
                        help="Buffer window size (default = 4)")
    parser.add_argument("-f", "--frameskip", type=int, default=4,
                        help="Frames skipped after each action (default = 4)")
    parser.add_argument("-dp", "--dopamine_pooling", action='store_true',
                        help="Enable dopamine-like frameskipping")
    parser.add_argument("-m", "--modifs", nargs="+",
                        default=[], help="List of modifications to apply")
    parser.add_argument("-rf", "--reward_function", type=str,
                        default="", help="Custom reward function path")
    parser.add_argument("-a", "--agents", nargs='+',
                        required=True, help="List of trained agent model paths")
    parser.add_argument("-mo", "--game_mode", type=int,
                        default=0, help="Alternative ALE game mode")
    parser.add_argument("-d", "--difficulty", type=int,
                        default=0, help="Alternative ALE difficulty")
    parser.add_argument("-e", "--episodes", type=int,
                        default=10, help="Number of episodes to run per agent")
    parser.add_argument("-wr", "--wrapper", type=str,
                        default="", help="Use a masking wrapper")

    args = parser.parse_args()

    # Initialize environment
    env = HackAtari(
        args.game,
        args.modifs,
        args.reward_function,
        dopamine_pooling=args.dopamine_pooling,
        game_mode=args.game_mode,
        difficulty=args.difficulty,
        render_mode="None",
        obs_mode=args.obs_mode,
        mode="ram",
        hud=False,
        render_oc_overlay=True,
        buffer_window_size=args.window,
        frameskip=args.frameskip,
        repeat_action_probability=0.25,
        full_action_space=False,
    )
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    if args.wrapper == "binary":
        env = BinaryMaskWrapper(env)
    elif args.wrapper == "object":
        env = PixelMaskWrapper(env)
    elif args.wrapper == "class":
        env = ObjectTypeMaskWrapper(env)
    elif args.wrapper == "planes":
        env = ObjectTypeMaskPlanesWrapper(env)

    avg_results = []
    std_results = []
    total_runs = []

    # Iterate through all agent models
    for agent_path in args.agents:
        agent, policy = load_agent(agent_path, env, "cpu")
        print(f"Loaded agent from {agent_path}")

        rewards = []
        for episode in range(args.episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = policy(torch.Tensor(obs).unsqueeze(0))[0]
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated

            rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward}")

        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        avg_results.append(avg_reward)
        std_results.append(std_reward)
        total_runs.append(args.episodes)

        print("\nSummary:")
        print(f"Agent: {agent_path}")
        print(f"Total Episodes: {args.episodes}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Standard Deviation: {std_reward:.2f}")
        print(f"Min Reward: {np.min(rewards)}")
        print(f"Max Reward: {np.max(rewards)}")
        print("--------------------------------------")

    # Compute overall statistics
    total_avg, total_std = combine_means_and_stds(
        avg_results, std_results, total_runs)
    print("------------------------------------------------")
    print(f"Overall Average Reward: {total_avg:.2f}")
    print(f"Overall Standard Deviation: {total_std:.2f}")
    print("------------------------------------------------")

    env.close()


if __name__ == "__main__":
    main()
