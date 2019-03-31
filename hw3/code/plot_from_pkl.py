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
    parser.add_argument('exp_name', type=str)
    args = parser.parse_args()

    pickle_filenames = [os.path.join(out_folder, args.exp_name, fn) for fn in os.listdir(os.path.join(out_folder, args.exp_name)) if re_pattern.search(fn) is not None]

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


            if (new_min_timesteps_value[-1] < min_timesteps_value) or (min_timesteps_value == -1):
                min_timesteps_value = new_min_timesteps_value[-1]
                min_timesteps = new_min_timesteps_value

    mean_100_rew = np.asarray([a[:(min_timesteps.shape[0])] for a in mean_100_rew])
    best_mean_rew = np.asarray([a[:(min_timesteps.shape[0])] for a in best_mean_rew])

    mean_100_rew_avg = np.mean(mean_100_rew, axis = 0)
    mean_100_rew_std = filter_infs(np.std(mean_100_rew, axis = 0))

    best_mean_rew_avg = np.mean(best_mean_rew, axis = 0)
    best_mean_rew_std = filter_infs(np.std(best_mean_rew, axis = 0))

    plt.errorbar(min_timesteps, mean_100_rew_avg, mean_100_rew_std, color='r', linewidth=2.0)

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylabel('Mean last 100 episode rew')
    plt.xlabel('Timesteps')    

    plt.savefig(os.path.join(out_folder, args.exp_name, 'mean_100_%s.png'%(args.exp_name)), bbox_inches='tight')

    plt.clf()

    plt.errorbar(min_timesteps, best_mean_rew_avg, best_mean_rew_std, color='r', linewidth=2.0)

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylabel('Best mean episode rew')
    plt.xlabel('Timesteps')    

    plt.savefig(os.path.join(out_folder, args.exp_name, 'best_mean_%s.png'%(args.exp_name)), bbox_inches='tight')

    plt.clf()    

    

if __name__ == "__main__":
    main()
