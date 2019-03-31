import os
import re
import shutil
import importlib



envnames = ['Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','Reacher-v2','Walker2d-v2']

n_rollouts = range(5,201,5)

for envname in envnames:
        os.system('cd code && python3 dagger.py %s --num_max_rollouts 200'%(envname))
        for n_rollout in n_rollouts:        
                # In case the user did not run experts already
                # os.system('cd code && python3 run_expert.py %s --num_rollouts %d --n_rollout_folder'%(envname, n_rollout))
                # os.system('cd code && python3 run_expert.py %s --num_rollouts %d --val --n_rollout_folder'%(envname,n_rollout))                
                os.system('cd code && python3 run_dagger.py %s --num_training_rollouts %d --num_rollouts 200 --render'%(envname, n_rollout))