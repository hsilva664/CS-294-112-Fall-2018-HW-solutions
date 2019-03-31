#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from global_defs import GlobalDefs

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--visible_device_list',type=str, default='1')
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')

    parser.add_argument('--n_rollout_folder', action='store_true',
                        help='Whether to write on separate rollouts folder (for varying rollout number)')                        

    args = parser.parse_args()

    expert_policy_file = os.path.join(GlobalDefs.expert_policy_folder, args.envname + '.pkl')

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(expert_policy_file)
    print('loaded and built')

    if args.val:
        num_rollouts = min(20,args.num_rollouts)
    else:
        num_rollouts = args.num_rollouts

    gpu_options = tf.GPUOptions(allow_growth=True,visible_device_list=args.visible_device_list)
    config = tf.ConfigProto(gpu_options=gpu_options)    

    with tf.Session(config=config):
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, d = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        env.close()

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        string_prefix = 'val_' if args.val else 'train_'

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions),
                       'returns': { 'values': returns, 'mean': np.mean(returns), 'std': np.std(returns)} }

        if args.n_rollout_folder == False:
            if not tf.gfile.Exists( GlobalDefs.expert_results_folder ) :  
                tf.gfile.MakeDirs(GlobalDefs.expert_results_folder)            
            with open(os.path.join(GlobalDefs.expert_results_folder, string_prefix + args.envname + '.pkl'), 'wb') as f:
                pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)
        else:
            experts_path = os.path.join(GlobalDefs.rollout_base, '%d'%(args.num_rollouts), GlobalDefs.expert_results_folder)
            if not tf.gfile.Exists( experts_path ) :  
                tf.gfile.MakeDirs(experts_path)
            with open(os.path.join( experts_path, string_prefix + args.envname + '.pkl'), 'wb') as f:
                pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)            

if __name__ == '__main__':
    main()
