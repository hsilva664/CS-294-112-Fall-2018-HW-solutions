import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

varying_rollout_folder = 'code/varying_rollout'
clone_results_subfolder = 'clone_results'

out_folder = '2_3_graphs'

envnames = ['Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','Reacher-v2','Walker2d-v2']
rollouts = [a for a in range(5,201,5)]

x = np.array( rollouts ,dtype=np.int32)

mean_dict = {}
std_dict = {}


for envname in envnames:
    mean_list = []
    std_list = []
    for rollout in rollouts:
        clone_results_fname = os.path.join(varying_rollout_folder, str(rollout), clone_results_subfolder, envname + '.pkl' ) 

        with open(clone_results_fname, 'rb') as f:
            clone_results = pickle.loads(f.read())   

        mean_list.append( clone_results['returns']['mean'] )
        std_list.append( clone_results['returns']['std'] )

    mean_dict[envname] = np.array(mean_list, dtype = np.float32)
    std_dict[envname] = np.array(std_list, dtype = np.float32)

    plt.errorbar(x, mean_dict[envname], std_dict[envname], color='r', linewidth=2.0)
    plt.ylabel('Reward')
    plt.xlabel('# rollouts')    

    plt.savefig(os.path.join(out_folder,'%s.png'%(envname)), bbox_inches='tight')

    plt.clf()
