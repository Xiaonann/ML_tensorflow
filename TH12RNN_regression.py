'''
use sin plot to predict cos?
'''

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

INPUT_SIZE = 1
OUTPUT_SIZE = 1
TIME_STEPS = 20
HIDDEN_NEURONS = 10

BATCH_SIZE = 50
BATCH_START = 0
BATCH_START_TEST = 0

learning_rate = 0.006

def get_batch():
    global BATCH_START, TIME_STEPS
    xs = np.arange(BATCH_START,BATCH_START+TIME_STEPS*BATCH_SIZE)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
#    plt.plot(xs[0,:],res[0,:],'r',xs[0,:],seq[0,:],'b--')
#    plt.show()  
    return [seq[:,:,np.newaxis],res[:,:,np.newaxis],xs]

class LSTMRNN(object):
    def __init__(self,n_steps,input_size,output_size,cell_size,batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope('input'):
            self.xs = tf.placeholder(tf.float32,[None,n_steps,input_size],name='xs')
            self.ys = tf.placeholder(tf.float32,[None,n_steps,output_size],name='ys')
        with tf.variable_scope('hidden_after_in'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('hidden_before_out'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
    
    def _weight_variable(self,shape,name='weights'):
        weight_init = tf.random_normal_initializer(mean=0,stddev=1.,)
        return tf.get_variable(shape=shape,initializer=weight_init,name=name)

    def _bias_variable(self,shape,name='biases'):
        bias_init = tf.constant_initializer(0.1)
        return tf.get_variable(shape=shape,initializer=bias_init,name=name)

    
    def add_input_layer(self,):
        X = tf.reshape(self.xs,[-1,self.n_input])
    #    weights = _weight
    #    hidden_in = tf.matmul(X,['in']) + biases['in'] #(128*28,n_neurons)
    #    hidden_in = tf.reshape(hidden_in,[-1,n_step,n_neurons])


            


