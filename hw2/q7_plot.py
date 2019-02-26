import os
import re

ll_pattern = re.compile('ll_.*')

base_folder = 'question_7/data'
out_folder = 'question_7'

ll_folders = [os.path.join(base_folder,a) for a in os.listdir(base_folder) if ll_pattern.search(a) is not None]

os.system('python3 plot_and_save.py %s --outname %s'%(' '.join(ll_folders), os.path.join(out_folder,'plot.png') ))