# OCAtariWrappers

This is the official repository for the paper _"Balancing Abstraction and Spatial Relationships for Robust
Reinforcement Learning"_.

This repository includes different wrappers to be used with [OCAtari](https://github.com/k4ntz/OC_Atari)
to generate different object-centric masked input representations.

## Install
``
pip install "gymnasium[atari, accept-rom-license]"
pip install -r requirements.txt
pip install .
``

## Usage
```python
from ocatari_wrappers import BinaryMaskWrapper
from ocatari import OCAtari

env = OCAtari("ALE/Frostbite")

env = BinaryMaskWrapper(env, include_pixels=True)

obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

## Test Setup
First test if the Backend is set up correctly
``
python scripts/run.py -g Pong -hu
``

Now we test if the wrappers are also set up
``
python scripts/print_state.py
``

If everything works as intended you should now have an svg showing you the binary mask in the game of Freeway after 100 steps.


## Citing
Please cite as stated.