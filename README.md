# OCAtariWrappers

This is the official repository for the paper _"Balancing Abstraction and Spatial Relationships for Robust
Reinforcement Learning"_.

This repository includes different wrappers to be used with [OCAtari](https://github.com/k4ntz/OC_Atari)
to generate different object-centric masked input representations.

## Install
``
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


## Citing
Please cite as stated.