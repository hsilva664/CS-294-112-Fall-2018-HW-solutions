import os
import re
import shutil
import importlib



envnames = ['Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','Reacher-v2','Walker2d-v2']

n_rollouts = range(5,201,5)

for envname in envnames:
        os.system('cd code && python3 dagger.py %s --num_max_rollouts 200'%(envname))
        for n_rollout in n_rollouts:        
                os.system('cd code && python3 run_dagger.py %s --num_training_rollouts %d --num_rollouts 200'%(envname, n_rollout))