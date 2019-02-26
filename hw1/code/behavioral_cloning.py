#!/usr/bin/env python

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
from model import fc_net
from tensorflow.python import debug as tf_debug
from global_defs import GlobalDefs

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--fixed_epochs', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--visible_device_list',type=str, default='1')
    parser.add_argument('--batch_size',type=int,default=1000)
    parser.add_argument('--learning_rate',type=float,default=1e-3)
    parser.add_argument('--num_training_rollouts',type=int)
    args = parser.parse_args()

    if not args.num_training_rollouts:
        EXPERT_TRAIN_FILE = os.path.join(GlobalDefs.expert_results_folder,'train_'+args.envname+'.pkl')
        EXPERT_VAL_FILE = os.path.join(GlobalDefs.expert_results_folder,'val_'+args.envname+'.pkl')
    else:
        experts_path = os.path.join(GlobalDefs.rollout_base, '%d'%(args.num_training_rollouts), GlobalDefs.expert_results_folder)
        EXPERT_TRAIN_FILE = os.path.join(experts_path,'train_'+args.envname+'.pkl')
        EXPERT_VAL_FILE = os.path.join(experts_path,'val_'+args.envname+'.pkl')
        

    with open(EXPERT_TRAIN_FILE, 'rb') as f:
        train_data = pickle.loads(f.read())

    with open(EXPERT_VAL_FILE, 'rb') as f:
        val_data = pickle.loads(f.read())        

    observations = train_data['observations']
    actions = train_data['actions']

    val_observations = val_data['observations']
    val_actions = val_data['actions']    

    # Training dataset
    with tf.name_scope("train_fetching"):
        train_dataset = tf.data.Dataset.from_tensor_slices((observations, actions))
        train_dataset = train_dataset.shuffle(buffer_size=10000)
        batched_train_dataset = train_dataset.batch(args.batch_size)

    # Validation dataset
    with tf.name_scope("eval_fetching"):
        val_dataset = tf.data.Dataset.from_tensor_slices((val_observations, val_actions))
        batched_val_dataset = val_dataset.batch(args.batch_size)

    #Training dataset
    with tf.name_scope("get_data"):
        iterator = tf.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                        batched_train_dataset.output_shapes)

        x, y = iterator.get_next()           

        train_init_op = iterator.make_initializer(batched_train_dataset)
        val_init_op = iterator.make_initializer(batched_val_dataset) 

    #Predict action
    with tf.name_scope("predict_action"):    
        is_training = tf.placeholder(tf.bool, shape = [] )
        net = fc_net(input_size = observations.shape[1], output_size=actions.shape[2])
        loss = net.predict(x,y, is_training)    
 

    with tf.name_scope("optimizer_details"):
        train_step = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)

    #Config
    gpu_options = tf.GPUOptions(allow_growth=True,visible_device_list=args.visible_device_list)
    config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=config)

    saver = tf.train.Saver()


    if args.num_training_rollouts == False:
        full_ckpt_dir = os.path.join(GlobalDefs.clone_models_folder, args.envname)
    else:
        full_ckpt_dir = os.path.join(GlobalDefs.rollout_base, '%d'%(args.num_training_rollouts),GlobalDefs.clone_models_folder, args.envname)

    if tf.gfile.Exists(full_ckpt_dir):
        tf.gfile.DeleteRecursively(full_ckpt_dir)    
    tf.gfile.MakeDirs(full_ckpt_dir)    
    full_ckpt_filename = os.path.join(full_ckpt_dir, 'checkpoint.ckpt')

    sess.run(tf.global_variables_initializer())

    EPOCHS = 300
    EPOCHS_TO_STOP = 50

    epoch = 0
    step = 0
    min_val_loss = 1000.
    early_stop_counter = 0
    

    with sess.as_default():
        if args.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        try:
            while True:
                print('Starting epoch %d / %d' % (epoch + 1, EPOCHS))
                epoch_loss = 0.
                epoch_steps = 0
                train_init_op.run()

                while True:
                    try:
                        _, batch_loss = sess.run([train_step, loss], feed_dict={is_training: True})
                        step += 1

                        print('Epoch #%d; Loss: %f;' % (epoch+1, batch_loss))
                        epoch_loss += batch_loss
                        epoch_steps += 1
                    except tf.errors.OutOfRangeError:
                        break

                avg_epoch_loss = epoch_loss/epoch_steps
                print('Average epoch loss: %f' % avg_epoch_loss)                

                val_init_op.run()
                val_epoch_loss = 0.
                val_epoch_steps = 0
                while True:
                    try:
                        batch_loss = sess.run(loss, feed_dict={is_training: False})
                        val_epoch_loss += batch_loss
                        val_epoch_steps += 1
                    except tf.errors.OutOfRangeError:
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
                    saver.save(sess, full_ckpt_filename)

                epoch += 1

        except KeyboardInterrupt:
            print('\nInterrupted at epoch ' + str(epoch))
            print('Average_loss = ' + str(avg_epoch_loss))
            print('Average val loss = ' + str(val_avg_epoch_loss))
            saver.save(sess, full_ckpt_filename) 
            print('Model saved')
            pass                         

        

if __name__ == '__main__':
    main()
