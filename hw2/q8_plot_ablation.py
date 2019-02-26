import os
import pandas as pd
import numpy as np
import json

parent_dir = 'question_8/part_2/data'
out_dir = 'question_8/part_2'
all_dirs = [ os.path.join(parent_dir,a) for a in os.listdir(parent_dir) if os.path.isdir( os.path.join(parent_dir,a) )]


os.system('python3 plot_and_save.py %s --outname %s'%(' '.join(all_dirs), os.path.join(out_dir,'all_plots.png') ))