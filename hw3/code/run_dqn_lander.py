import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from multiprocessing import Process
from tensorflow.python import debug as tf_debug

import dqn
from dqn_utils import *

def lander_model(obs, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = obs
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out

def lander_optimizer():
    return dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        lr_schedule=ConstantSchedule(1e-3),
        kwargs={}
    )

def lander_stopping_criterion(num_timesteps):
    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps
    return stopping_criterion

def lander_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 1),
            (num_timesteps * 0.1, 0.02),
        ], outside_value=0.02
    )

def lander_kwargs():
    return {
        'optimizer_spec': lander_optimizer(),
        'q_func': lander_model,
        'replay_buffer_size': 50000,
        'batch_size': 32,
        'gamma': 1.00,
        'learning_starts': 1000,
        'learning_freq': 1,
        'frame_history_len': 1,
        'target_update_freq': 3000,
        'grad_norm_clipping': 10,
        'lander': True
    }

def lander_learn(env,
                 session,
                 seed,
                 exp_name,
                 num_timesteps,
                 double_q):

    optimizer = lander_optimizer()
    stopping_criterion = lander_stopping_criterion(num_timesteps)
    exploration_schedule = lander_exploration_schedule(num_timesteps)

    dqn.learn(
        env=env,
        session=session,
        exp_name=exp_name,
        seed = seed,         
        exploration=lander_exploration_schedule(num_timesteps),
        stopping_criterion=lander_stopping_criterion(num_timesteps),
        double_q=double_q,
        **lander_kwargs()
    )
    env.close()

def set_global_seeds(i):
    tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def get_session(visible_gpus, debug):
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(allow_growth=True,visible_device_list=visible_gpus) #Other GPU in use
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
        gpu_options=gpu_options)
    # GPUs don't significantly speed up deep Q-learning for lunar lander,
    # since the observations are low-dimensional
    session = tf.Session(config=tf_config)
    if debug:
        session = tf_debug.LocalCLIDebugWrapperSession(session)        
    return session

def get_env(seed):
    env = gym.make('LunarLander-v2')

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/hw3_vid_dir/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)

    return env

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--n_processes', type=int, default=1)
    parser.add_argument('--debug', action='store_true')    
    parser.add_argument('--visible_gpus', type=str, default='0')
    parser.add_argument('--double_q', action='store_true')
    args = parser.parse_args()

    processes = []
    seeds = []

    def single_learning_process(seed):
        # Run training
        print('random seed = %d' % seed)

        # Run training
        env = get_env(seed)
        session = get_session(args.visible_gpus, args.debug)
        lander_learn(env, session, seed, args.exp_name, num_timesteps=500000, double_q = args.double_q)

    if args.debug == True:
        seed = random.randint(0, 9999)
        single_learning_process(seed)
    else:
        for e in range(args.n_processes):   

            seed = random.randint(0, 9999)
            if seed not in seeds:
                seeds.append(seed)       
            else:
                while seed in seeds:
                    seed = random.randint(0, 9999)
                seeds.append(seed)

            proc = Process(target=single_learning_process, args=(seed,) )
            proc.start()
            processes.append(proc)

        for p in processes:
            p.join()             

if __name__ == "__main__":
    main()
