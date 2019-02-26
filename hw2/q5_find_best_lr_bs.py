import os
import pandas as pd
import numpy as np
import json

parent_dir = 'question_5/data'
all_dirs = [os.path.join(parent_dir,a) for a in os.listdir(parent_dir) if os.path.isdir( os.path.join(parent_dir,a) )]

passing_lr = []
passing_bs = []
passing_tuples = []

max_mean = 0.0

for s_dir in all_dirs:
        exp_dirs = [os.path.join(s_dir,a) for a in os.listdir(s_dir) if os.path.isdir( os.path.join(s_dir,a) )]

        whole_experiment_data = np.array([0.])

        skip = False
        for exp_dir in exp_dirs:
                log_path = os.path.join(exp_dir,'log.txt')
                experiment_data = pd.read_table(log_path) 
                experiment_data = np.array(experiment_data['AverageReturn'])

                if experiment_data.shape[0] < 100:
                        skip = True
                        break

                whole_experiment_data = whole_experiment_data + ( experiment_data / float( len(exp_dirs) ) )
        if skip:
                continue

        last_iter_mean = np.mean( whole_experiment_data[90:] )
        if last_iter_mean > max_mean:
                max_mean = last_iter_mean
        if last_iter_mean >= 700:
                param_path = open(os.path.join(exp_dirs[0],'params.json'))
                params = json.load(param_path)
                lr = params['learning_rate']
                bs = params['min_timesteps_per_batch']
                passing_lr.append(lr)
                passing_bs.append(bs)
                passing_tuples.append((lr,bs))

print(max_mean)

max_lr = max(passing_lr)
min_bs = min(passing_bs)

if (max_lr,min_bs) in passing_tuples:
        print( (max_lr,min_bs) )     
else:
        print('Max lr %f'%(max_lr))
        print('Min bs %d'%(min_bs))

        print('Passing tuples:')
        print(passing_tuples)