# A small example how to run the eval script in the default environment, in a perturbation environment, and with different observation wrappers. Other games follow the same pattern. The exact names of the modifications can be found in the paper or under HackAtari. 

python scripts/eval.py -g Pong -a models/Pong/0/ppo.cleanrl_model
python scripts/eval.py -g Pong -a models/Pong/0/ppo.cleanrl_model -m lazy_enemy

python scripts/eval.py -g Pong -a models/Pong/0/obj_vector.cleanrl_model -obs obj
python scripts/eval.py -g Pong -a models/Pong/0/obj_vector.cleanrl_model -obs obj -m lazy_enemy

python scripts/eval.py -g Pong -a models/Pong/0/ppo_object.cleanrl_model -wr pixels
python scripts/eval.py -g Pong -a models/Pong/0/ppo_object.cleanrl_model -wr pixels -m lazy_enemy

python scripts/eval.py -g Pong -a models/Pong/0/ppo_binary.cleanrl_model -wr binary
python scripts/eval.py -g Pong -a models/Pong/0/ppo_binary.cleanrl_model -wr binary -m lazy_enemy

python scripts/eval.py -g Pong -a models/Pong/0/ppo_classes.cleanrl_model -wr classes
python scripts/eval.py -g Pong -a models/Pong/0/ppo_classes.cleanrl_model -wr classes -m lazy_enemy

python scripts/eval.py -g Pong -a models/Pong/0/ppo_planes.cleanrl_model -wr planes
python scripts/eval.py -g Pong -a models/Pong/0/ppo_planes.cleanrl_model -wr planes -m lazy_enemy