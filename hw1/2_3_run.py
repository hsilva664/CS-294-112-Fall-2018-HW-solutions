import os
import re
import shutil
import importlib



envnames = ['Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','Reacher-v2','Walker2d-v2']
n_rollouts = range(5,201,5)

for envname in envnames:
    for n_rollout in n_rollouts:
        os.system('cd code && python3 run_expert.py %s --num_rollouts %d --n_rollout_folder'%(envname, n_rollout))
        os.system('cd code && python3 run_expert.py %s --num_rollouts %d --val --n_rollout_folder'%(envname,n_rollout))
        os.system('cd code && python3 behavioral_cloning.py %s --num_training_rollouts %d'%(envname, n_rollout))
        os.system('cd code && python3 run_clone.py %s --num_training_rollouts %d --num_rollouts 200'%(envname, n_rollout))
