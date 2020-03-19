import tensorflow as tf 
import numpy as np 


### hidden layer function
def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    # use normal distribution function to generate a 
    #  random variables matrix better than zeros
    layer_name = 'layer%s'%n_layer
    with tf.name_scope('hiddenlayer1'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]))
            tf.histogram_summary(layer_name+'weights',Weights)
        with tf.name_scope('Biases'):
        # bias like a list recommend not 0
            biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
        with tf.name_scope('Learningfunction'):
            learning_fun = tf.matmul(tf.cast(inputs,tf.float32),Weights) + biases
        if activation_function is None:
            outputs = learning_fun
        else:
            outputs = activation_function(learning_fun)
        return outputs
###


### define input layer
x_real = np.linspace(-1,1,300)[:,np.newaxis] #[:,np.newaxis] dimension
# add a noise (mean,standard deviation,dimension) more close to real
noise = np.random.normal(0,0.05,x_real.shape)
y_real = np.square(x_real) - 0.5 + noise
# define placeholder
# using plaxeholder can decide the size of training data sometimes speed up
with tf.name_scope('inputlayer'):
    x_ph = tf.placeholder(tf.float32,[None,1],name='x_input')
    y_ph = tf.placeholder(tf.float32,[None,1],name='y_input')
###

### design a hidden layer
# has 10 neurons(input to hidden1:1 to 10)
# input is one atrribution(only x) seen as one neuron = output
hidden_layer1 = add_layer(x_ph,1,10,n_layer = 1,activation_function=tf.nn.relu)
###

### design output layer
y_predict = add_layer(hidden_layer1,10,1,n_layer = 2,activation_function=None)
###

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y_ph-y_predict))

    optimizier = tf.train.GradientDescentOptimizer(0.2)
with tf.name_scope('training'):
    train_function = optimizier.minimize(loss)

init = tf.global_variables_initializer()


sess = tf.Session()
# write the structure into a file in the folder logs
writer = tf.summary.FileWriter("logs/",sess.graph)
sess.run(init)


