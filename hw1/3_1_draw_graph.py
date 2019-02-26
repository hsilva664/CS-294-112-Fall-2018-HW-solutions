import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

varying_rollout_folder = 'code/varying_rollout'
clone_results_subfolder = 'clone_results'
dagger_results_subfolder = 'dagger_results'
expert_results_folder = 'code/expert_results'


out_folder = '3_1_graphs'

envnames = ['Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','Reacher-v2','Walker2d-v2']
rollouts = [a for a in range(5,201,5)]

x = np.array( rollouts ,dtype=np.int32)

c_mean_dict = {}
c_std_dict = {}

d_mean_dict = {}
d_std_dict = {}

for envname in envnames:
    c_mean_list = []
    c_std_list = []
    d_mean_list = []
    d_std_list = []    
    for rollout in rollouts:
        clone_results_fname = os.path.join(varying_rollout_folder, str(rollout), clone_results_subfolder, envname + '.pkl' ) 
        dagger_results_fname = os.path.join(varying_rollout_folder, str(rollout), dagger_results_subfolder, envname + '.pkl' ) 

        with open(clone_results_fname, 'rb') as f:
            clone_results = pickle.loads(f.read()) 

        with open(dagger_results_fname, 'rb') as f:
            dagger_results = pickle.loads(f.read())               

        c_mean_list.append( clone_results['returns']['mean'] )
        c_std_list.append( clone_results['returns']['std'] )

        d_mean_list.append( dagger_results['returns']['mean'] )
        d_std_list.append( dagger_results['returns']['std'] )        
            

    c_mean_dict[envname] = np.array(c_mean_list, dtype = np.float32)
    c_std_dict[envname] = np.array(c_std_list, dtype = np.float32)

    d_mean_dict[envname] = np.array(d_mean_list, dtype = np.float32)
    d_std_dict[envname] = np.array(d_std_list, dtype = np.float32)    

    expert_results_fname = os.path.join(expert_results_folder, 'train_' + envname +'.pkl')  
    with open(expert_results_fname, 'rb') as f:
        expert_results = pickle.loads(f.read()) 
    e_mean = np.array(expert_results['returns']['mean'])

    lines0 = plt.errorbar(x, c_mean_dict[envname], c_std_dict[envname],color='r', linewidth=2.0, label='Behavioral Cloning')

    lines1 = plt.errorbar(x, d_mean_dict[envname], d_std_dict[envname], color='g', linewidth=2.0, label='Dagger')

    lines2 = plt.plot(x, e_mean * np.ones_like(d_mean_dict[envname]) )
    plt.setp(lines2, color='b', linewidth=2.0, label='Expert')        

    plt.ylabel('Reward')
    plt.xlabel('# rollouts/ Dagger iterations')    

    plt.legend()

    plt.savefig(os.path.join(out_folder,'%s.png'%(envname)), bbox_inches='tight')

    plt.clf()
