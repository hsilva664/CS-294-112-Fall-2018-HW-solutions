import os
from multiprocessing import Process, Lock
import re

base_folder = 'code/data'
data_folder = 'data'

system_code = [ 'cd code && python3 train_ac_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.95 -n 100 -l 2 -s 64 -b 5000 -e 3 -lr 0.01 --exp_name 10_10 -ntu 10 -ngsptu 10 --visible_gpus 0', \
                'cd code && python3 train_ac_f18.py HalfCheetah-v2 -ep 150 --discount 0.90 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 --exp_name 10_10 -ntu 10 -ngsptu 10 --visible_gpus 1']

def run_funct( code):
    os.system(code)

processes = []

for idx, single_code in enumerate(system_code):
    p = Process(target=run_funct, args=(single_code,) )
    p.start()
    processes.append(p)    

for p in processes:
    p.join()

env_names = ['InvertedPendulum-v2', 'HalfCheetah-v2']

folder_names = []
for env_name in env_names:
    pattern = re.compile('ac_10_10_' + env_name + '.*')
    matching_folders = [os.path.join(data_folder,a) for a in os.listdir(base_folder) if pattern.search(a) is not None]
    print('Plotting %s'%(env_name))
    os.system('cd code && python3 plot.py %s'%(matching_folders[0]))


