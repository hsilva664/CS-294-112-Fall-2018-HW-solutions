import pickle
import os
from prettytable import PrettyTable

clone_results_folder = 'code/clone_results'
expert_results_folder = 'code/expert_results'

x = PrettyTable()

envnames = ['Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','Reacher-v2','Walker2d-v2']

x.field_names = ['Activities', 'Behavioral cloning', 'Expert']


for envname in envnames:
    clone_results_fname = os.path.join(clone_results_folder, envname +'.pkl') 
    expert_results_fname = os.path.join(expert_results_folder, 'train_' + envname +'.pkl')

    with open(clone_results_fname, 'rb') as f:
        clone_results = pickle.loads(f.read())   

    with open(expert_results_fname, 'rb') as f:
        expert_results = pickle.loads(f.read())   

    bc = str(clone_results['returns']['mean']) + '+-' + str(clone_results['returns']['std'])
    exp = str(expert_results['returns']['mean']) + '+-' + str(expert_results['returns']['std'])

    x.add_row([envname, bc, exp])


print(x)