import os
from multiprocessing import Process, Lock
import re

base_folder = 'code/data'
data_folder = 'data'



exp_names = ['rec_hist_60_delta_3','rec_hist_60_delta_5','rec_hist_60_delta_10']

system_code =  ['cd code && python3 train_policy.py pm --exp_name q3_%s -e 3 --history 60 --discount 0.90 -lr 5e-4 -n 100 --recurrent -s 64 -rs 32 --disjoint_sets --delta 3',
                'cd code && python3 train_policy.py pm --exp_name q3_%s -e 3 --history 60 --discount 0.90 -lr 5e-4 -n 100 --recurrent -s 64 -rs 32 --disjoint_sets --delta 5',
                'cd code && python3 train_policy.py pm --exp_name q3_%s -e 3 --history 60 --discount 0.90 -lr 5e-4 -n 100 --recurrent -s 64 -rs 32 --disjoint_sets --delta 10']


system_code = [s%(i) for s,i in zip(system_code,exp_names)]

def run_funct(lock, code, gpu):
    lock.acquire()
    append_str = " --visible_gpus %d"%(gpu)
    os.system(code + append_str)
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
    pattern = re.compile('q3_' + exp_name + '_.*')
    matching_folders = [os.path.join(data_folder,a) for a in os.listdir(base_folder) if pattern.search(a) is not None]
    folder_names.append(matching_folders[0])


os.system('cd code && python3 plot_train_vs_eval.py %s --legend %s'%(' '.join(folder_names), ' '.join(exp_names) ) )