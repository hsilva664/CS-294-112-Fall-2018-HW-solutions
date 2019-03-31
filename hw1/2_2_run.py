import os
import re
import shutil
import importlib



envnames = ['Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','Reacher-v2','Walker2d-v2']


for envname in envnames:
    os.system('cd code && python3 run_expert.py %s --num_rollouts 200'%(envname))
    os.system('cd code && python3 run_expert.py %s --num_rollouts 20 --val'%(envname))
    os.system('cd code && python3 behavioral_cloning.py %s'%(envname))
    os.system('cd code && python3 run_clone.py %s --num_rollouts 200'%(envname))
