import os
from multiprocessing import Process, Lock
import re

base_folder = 'code/data'
data_folder = 'data'
exp_names = ['1_1','100_1','1_100','10_10']

system_code = [ 'cd code && python3 train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name %s -ntu 1 -ngsptu 1 --visible_gpus 0'%(exp_names[0]), \
                'cd code && python3 train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name %s -ntu 100 -ngsptu 1 --visible_gpus 1'%(exp_names[1]), \
                'cd code && python3 train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name %s -ntu 1 -ngsptu 100 --visible_gpus 0'%(exp_names[2]), \
                'cd code && python3 train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name %s -ntu 10 -ngsptu 10 --visible_gpus 1'%(exp_names[3])]

def run_funct(lock, code):
    lock.acquire()
    os.system(code)
    lock.release()

processes = []
locks = [Lock() for _ in range(2)]

for idx, single_code in enumerate(system_code):
    p = Process(target=run_funct, args=(locks[idx % len(locks)], single_code))
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

