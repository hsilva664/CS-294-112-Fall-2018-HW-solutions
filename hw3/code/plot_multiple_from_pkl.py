import pickle
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import re

out_folder = 'plots'
re_pattern = re.compile('.*\.pkl')

def filter_infs(np_arr):
    np_arr[np.logical_not(np.isfinite(np_arr))] = 0
    return np_arr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_name', type=str)    
    parser.add_argument('exp_names', type=str, nargs='*')    
    args = parser.parse_args()

    all_mean_100_rew_avg = []
    all_mean_100_rew_std = []

    all_best_mean_rew_avg = []
    all_best_mean_rew_std = []

    global_min_timesteps = None
    global_min_timesteps_value = -1

    for exp_name_i in args.exp_names:

        pickle_filenames = [os.path.join(out_folder, exp_name_i, fn) for fn in os.listdir(os.path.join(out_folder, exp_name_i)) if re_pattern.search(fn) is not None]

        mean_100_rew = []
        best_mean_rew = []
        min_timesteps = None
        min_timesteps_value = -1

        for filename in pickle_filenames:
            with open(filename, 'rb') as f:
                pk_obj = pickle.load(f)
                new_mean_100_rew = pk_obj['mean_100_rew']
                new_best_mean_rew = pk_obj['best_mean_rew']
                new_min_timesteps_value = pk_obj['timesteps']
                mean_100_rew.append(new_mean_100_rew)
                best_mean_rew.append(new_best_mean_rew)

                if (new_min_timesteps_value[-1] < global_min_timesteps_value) or (global_min_timesteps_value == -1):
                    global_min_timesteps_value = new_min_timesteps_value[-1]
                    global_min_timesteps = new_min_timesteps_value

                if (new_min_timesteps_value[-1] < min_timesteps_value) or (min_timesteps_value == -1):
                    min_timesteps_value = new_min_timesteps_value[-1]
                    min_timesteps = new_min_timesteps_value

        mean_100_rew = np.asarray([a[:(min_timesteps.shape[0])] for a in mean_100_rew])
        best_mean_rew = np.asarray([a[:(min_timesteps.shape[0])] for a in best_mean_rew])

        all_mean_100_rew_avg.append(np.mean(mean_100_rew, axis = 0))
        all_mean_100_rew_std.append(filter_infs(np.std(mean_100_rew, axis = 0)))

        all_best_mean_rew_avg.append(np.mean(best_mean_rew, axis = 0))
        all_best_mean_rew_std.append(filter_infs(np.std(best_mean_rew, axis = 0)))

    colors = ['r','g','b','c','m','y','k']

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    for i in range(len(args.exp_names)):
        plt.plot(global_min_timesteps, all_mean_100_rew_avg[i][:(global_min_timesteps.shape[0])], \
                    color=colors[i % len(colors)], linewidth=2.0, label=args.exp_names[i])

        plt.ylabel('Mean last 100 episode rew')
        plt.xlabel('Timesteps')    

    plt.legend()
    plt.savefig(os.path.join(out_folder, 'mean_100_%s.png'%(args.out_name)), bbox_inches='tight')

    plt.clf()

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    for i in range(len(args.exp_names)):
        plt.plot(global_min_timesteps, all_best_mean_rew_avg[i][:(global_min_timesteps.shape[0])], \
                    color=colors[i % len(colors)], linewidth=2.0, label=args.exp_names[i])
        
        plt.ylabel('Best mean episode rew')
        plt.xlabel('Timesteps')    

    plt.legend()
    plt.savefig(os.path.join(out_folder, 'best_mean_%s.png'%(args.out_name)), bbox_inches='tight')

    plt.clf()    

    

if __name__ == "__main__":
    main()
