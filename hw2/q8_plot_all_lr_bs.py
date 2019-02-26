import os
import pandas as pd
import numpy as np
import json

parent_dir = 'question_8/part_1/data'
out_dir = 'question_8/part_1'
all_dirs_and_names = [ (os.path.join(parent_dir,a),a) for a in os.listdir(parent_dir) if os.path.isdir( os.path.join(parent_dir,a) )]


for s_dir, s_name in all_dirs_and_names:
        os.system('python3 plot_and_save.py %s --outname %s'%(s_dir, os.path.join(out_dir,s_name+'.png') ))