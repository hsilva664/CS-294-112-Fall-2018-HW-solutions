#!/usr/bin/env python

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
from model import fc_net
from global_defs import GlobalDefs

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--visible_device_list',type=str, default='0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--num_training_rollouts',type=int)                        
    args = parser.parse_args()


    #Import files to get dimension info
    EXPERT_TRAIN_FILE = os.path.join(GlobalDefs.expert_results_folder,'train_'+args.envname+'.pkl')

    with open(EXPERT_TRAIN_FILE, 'rb') as f:
        train_data = pickle.loads(f.read())    

    observations = train_data['observations']
    actions = train_data['actions']

    #Model checkpoint
    full_ckpt_dir = os.path.join(GlobalDefs.rollout_base, '%d'%(args.num_training_rollouts),GlobalDefs.dagger_models_folder, args.envname)
    full_ckpt_filename = os.path.join(full_ckpt_dir, 'checkpoint.ckpt')    

    x = tf.placeholder(tf.float64, shape = [1] + list(observations.shape[1:]) )

    #Predict action
    with tf.name_scope("predict_action"):    
        is_training = tf.placeholder(tf.bool, shape = [] )
        net = fc_net(input_size = observations.shape[1], output_size=actions.shape[2])
        pred = net.apply(x, is_training)    

    #Config
    gpu_options = tf.GPUOptions(allow_growth=True,visible_device_list=args.visible_device_list)
    config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=config)

    saver = tf.train.Saver()

    saver.restore(sess, full_ckpt_filename)

    with sess.as_default():

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = sess.run(pred, feed_dict={x: obs[None,:], is_training: False})
                observations.append(obs)
                actions.append(action[0])
                obs, r, done, _ = env.step(action[0])
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        beh_data = {'observations': np.array(observations),
                       'actions': np.array(actions),
                       'returns': { 'values': returns, 'mean': np.mean(returns), 'std': np.std(returns)} }


        base_out_dir = os.path.join(GlobalDefs.rollout_base, '%d'%(args.num_training_rollouts),GlobalDefs.dagger_results_folder)

        if not tf.gfile.Exists( base_out_dir ) :  
            tf.gfile.MakeDirs(base_out_dir)

        with open(os.path.join(base_out_dir, args.envname + '.pkl'), 'wb') as f:
            pickle.dump(beh_data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
