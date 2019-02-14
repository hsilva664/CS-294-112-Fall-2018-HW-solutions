import numpy as np
import tensorflow as tf
import random
import math

class fc_net(object):  


    def __init__(self, input_size, output_size):

        self.hidden_dim = 64

        self.input_size = input_size
        self.output_size = output_size

        #Convolutional network parameters
        self.n_hidden = 2

    def apply(self, x, is_training):
        with tf.variable_scope('main_network'):
            for i in range(self.n_hidden):
                x = tf.layers.dense(x, self.hidden_dim, \
                                    activation = tf.nn.tanh, \
                                    use_bias=True, \
                                    kernel_initializer=tf.keras.initializers.he_normal(), \
                                    bias_initializer=tf.zeros_initializer)

            x = tf.layers.dense(x, self.output_size, \
                                use_bias=True, \
                                kernel_initializer=tf.keras.initializers.he_normal(), \
                                bias_initializer=tf.zeros_initializer)       

        x = tf.expand_dims(x,1)
        return x

    def predict(self, x, y, is_training):

        x = self.apply(x, is_training)
        mse_loss = tf.losses.mean_squared_error(x, y)

        return mse_loss
