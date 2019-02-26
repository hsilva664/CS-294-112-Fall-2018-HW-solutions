#!/usr/bin/env python

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
from model import fc_net
from tensorflow.python import debug as tf_debug
from global_defs import GlobalDefs
import load_policy

class GetBatch(object):
    def __init__(self, obs, act, shuffle, batch_size):
        self.indices = np.random.randint(0, obs.shape[0], size = obs.shape[0])
        if shuffle:
            np.random.shuffle(self.indices)

        self.initial_index = 0
        self.batch_size = batch_size

        self.obs = obs
        self.act = act

    def get(self):
        ending_idx = min( self.obs.shape[0], self.initial_index + self.batch_size )

        out_obs = self.obs[self.initial_index:ending_idx, ...]
        out_act = self.act[self.initial_index:ending_idx, ...]

        self.initial_index = ending_idx

        return out_obs, out_act

    def done(self):
        return self.initial_index == self.obs.shape[0]
    


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--fixed_epochs', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--visible_device_list',type=str, default='0')
    parser.add_argument('--batch_size',type=int,default=1000)
    parser.add_argument('--learning_rate',type=float,default=1e-3)
    parser.add_argument('--num_max_rollouts',type=int)
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    ####Initial rollout
    expert_policy_file = os.path.join(GlobalDefs.expert_policy_folder, args.envname + '.pkl')
    policy_fn = load_policy.load_policy(expert_policy_file)

    #Config
    gpu_options = tf.GPUOptions(allow_growth=True,visible_device_list=args.visible_device_list)
    config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=config)

    with sess.as_default():
        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        observations = []
        actions = []

        obs = env.reset()
        done = False
        steps = 0
        while not done:
            action = policy_fn(obs[None,:])
            observations.append(obs)
            actions.append(action)
            obs, _, done, _ = env.step(action)
            steps += 1
            if args.render:
                env.render()
            if steps % 100 == 0: print("Initial policy: %i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break

    train_observations = np.array(observations)
    train_actions = np.array(actions)

    ####Initial rollout END


    ###Dagger preparation
    EXPERT_VAL_FILE = os.path.join(GlobalDefs.expert_results_folder,'val_'+args.envname+'.pkl')

    with open(EXPERT_VAL_FILE, 'rb') as f:
        val_data = pickle.loads(f.read())        

    val_observations = val_data['observations']
    val_actions = val_data['actions']    

    x = tf.placeholder(tf.float64, shape = [None] + list(train_observations.shape[1:]) )
    y = tf.placeholder(tf.float64, shape = [None] + list(train_actions.shape[1:]) )

    #Predict action
    with tf.name_scope("predict_action"):    
        is_training = tf.placeholder(tf.bool, shape = [] )
        net = fc_net(input_size = train_observations.shape[1], output_size=train_actions.shape[2])
        loss = net.predict(x,y, is_training)  
        pred = net.apply(x, is_training)    
 

    with tf.name_scope("optimizer_details"):
        train_step = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)

    saver = tf.train.Saver(max_to_keep=args.num_max_rollouts//5)

    sess.run(tf.global_variables_initializer())

    EPOCHS = 300
    EPOCHS_TO_STOP = 50

    recorded_rollouts = range(5,args.num_max_rollouts+1,5)


    for curr_rollout in range(1, (args.num_max_rollouts+1) ):
        epoch = 0
        step = 0
        min_val_loss = 1000.
        early_stop_counter = 0

        if curr_rollout in recorded_rollouts:
            full_ckpt_dir = os.path.join(GlobalDefs.rollout_base, '%d'%(curr_rollout),GlobalDefs.dagger_models_folder, args.envname)        
            if tf.gfile.Exists(full_ckpt_dir):
                tf.gfile.DeleteRecursively(full_ckpt_dir)    
            tf.gfile.MakeDirs(full_ckpt_dir)    
            full_ckpt_filename = os.path.join(full_ckpt_dir, 'checkpoint.ckpt')            
        
        with sess.as_default():
            if args.debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            ####Training model
            while True:
                print('Rollout %d: Starting epoch %d / %d' % (curr_rollout, epoch + 1, EPOCHS))
                epoch_loss = 0.
                epoch_steps = 0


                train_bg = GetBatch(train_observations, train_actions, True, args.batch_size)

                while True:

                    obs, act = train_bg.get()

                    _, batch_loss = sess.run([train_step, loss], feed_dict={is_training: True, x: obs, y: act })
                    step += 1

                    print('Rollout %d: Epoch #%d; Loss: %f;' % (curr_rollout, epoch+1, batch_loss))
                    epoch_loss += batch_loss
                    epoch_steps += 1

                    if train_bg.done():
                        break


                avg_epoch_loss = epoch_loss/epoch_steps
                print('Average epoch loss: %f' % avg_epoch_loss)                

                max_val_idx = min(val_observations.shape[0], curr_rollout*max_steps)

                val_bg = GetBatch(val_observations[:max_val_idx,...], val_actions[:max_val_idx,...], False, args.batch_size)
                val_epoch_loss = 0.
                val_epoch_steps = 0
                while True:
                    obs, act = val_bg.get()
                    batch_loss = sess.run(loss, feed_dict={is_training: False, x: obs, y: act})
                    val_epoch_loss += batch_loss
                    val_epoch_steps += 1

                    if val_bg.done():
                        break                        


                val_avg_epoch_loss = val_epoch_loss/val_epoch_steps
                print('Val average epoch loss: %f' % val_avg_epoch_loss)


                if (args.fixed_epochs == True) and (epoch == EPOCHS):
                    break
                elif args.fixed_epochs == False:                                   
                    dif = (min_val_loss - val_avg_epoch_loss)/abs(min_val_loss + 1e-6)
                    if dif < 0.05:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                    if early_stop_counter == EPOCHS_TO_STOP:
                        break

                if min_val_loss > val_avg_epoch_loss:
                    min_val_loss = val_avg_epoch_loss
                    if curr_rollout in recorded_rollouts:
                        saver.save(sess, full_ckpt_filename)

                epoch += 1

            ####Training model END


            ####Generating rollout from trained model
            import gym
            env = gym.make(args.envname)

            observations = []
            expert_actions = []

            obs = env.reset()
            done = False
            steps = 0
            while not done:
                model_action = sess.run(pred, feed_dict={x: obs[None,:], is_training: False})
                expert_action = policy_fn(obs[None,:])

                observations.append(obs)
                expert_actions.append(expert_action)

                obs, _, done, _ = env.step(model_action[0])
                steps += 1
                if steps % 100 == 0: print("Rollout %d; Generating data from trained model; %i/%i"%(curr_rollout, steps, max_steps))
                if steps >= max_steps:
                    break

            observations = np.array(observations)
            expert_actions = np.array(expert_actions)

            train_observations = np.concatenate( (train_observations, observations), axis = 0 )
            train_actions = np.concatenate( (train_actions, expert_actions), axis = 0 )
                     

        

if __name__ == '__main__':
    main()
