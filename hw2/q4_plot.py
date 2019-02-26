import os
import re

sb_pattern = re.compile('sb_.*')
lb_pattern = re.compile('lb_.*')

base_folder = 'question_4/data'
out_folder = 'question_4'

sb_folders = [os.path.join(base_folder,a) for a in os.listdir(base_folder) if sb_pattern.search(a) is not None]
lb_folders = [os.path.join(base_folder,a) for a in os.listdir(base_folder) if lb_pattern.search(a) is not None]

os.system('python3 plot_and_save.py %s --outname %s'%(' '.join(sb_folders), os.path.join(out_folder,'sb.png') ))
os.system('python3 plot_and_save.py %s --outname %s'%(' '.join(lb_folders), os.path.join(out_folder,'lb.png') ))