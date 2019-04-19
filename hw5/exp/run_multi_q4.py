import os
from multiprocessing import Process, Lock
import re

base_folder = 'code/data'
data_folder = 'data'

exp_names = ['HC_bc0', 'HC_bc0.001_kl0.1_dlr0.005_dti1000', 'HC_bc0.0001_kl0.1_dlr0.005_dti10000']

system_code = [ 'cd code && python3 train_ac_exploration_f18.py HalfCheetah-v2 -ep 150 --visible_gpus %d --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 --density_model none --exp_name HC_bc0', \
                'cd code && python3 train_ac_exploration_f18.py HalfCheetah-v2 -ep 150 --visible_gpus %d --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 --density_model ex2 -bc 0.001 -kl 0.1 -dlr 0.005 -dti 1000 --exp_name HC_bc0.001_kl0.1_dlr0.005_dti1000', \
                'cd code && python3 train_ac_exploration_f18.py HalfCheetah-v2 -ep 150 --visible_gpus %d --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 --density_model ex2 -bc 0.0001 -kl 0.1 -dlr 0.005 -dti 10000 --exp_name HC_bc0.0001_kl0.1_dlr0.005_dti10000']

def run_funct(lock, code, gpu):
    lock.acquire()
    os.system(code%(gpu))
    lock.release()

processes = []
locks = [Lock() for _ in range(2)]

for idx, single_code in enumerate(system_code):
    p = Process(target=run_funct, args=(locks[idx % len(locks)], single_code, idx % len(locks)))
    p.start()
    processes.append(p)    

for p in processes:
    p.join()


folder_names = []
for exp_name in exp_names:
    pattern = re.compile('ac_' + exp_name + '.*')
    matching_folders = [os.path.join(data_folder,a) for a in os.listdir(base_folder) if pattern.search(a) is not None]
    folder_names.append(matching_folders[0])


os.system('cd code && python3 plot.py %s --legend %s'%(' '.join(folder_names),' '.join(exp_names)))