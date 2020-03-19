# Normal types RELU sigmoid tanh 
# can design by self must be differentiable
# CNN use RELU, recurrent network RELU or tanh
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

### hidden layer function
def add_layer(inputs,in_size,out_size,activation_function=None):
    # use normal distribution function to generate a 
    #  random variables matrix better than zeros
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    # bias like a list recommend not 0
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    learning_fun = tf.matmul(tf.cast(inputs,tf.float32),Weights) + biases
    if activation_function is None:
        outputs = learning_fun
        #outputs = y_ph
    else:
        outputs = activation_function(learning_fun)
        #outputs = activation_function(y_ph)
    return outputs
###


### define input layer
x_real = np.linspace(-1,1,300)[:,np.newaxis] #[:,np.newaxis] dimension
# add a noise (mean,standard deviation,dimension) more close to real
noise = np.random.normal(0,0.05,x_real.shape)
y_real = np.square(x_real) - 0.5 + noise
# define placeholder
# using plaxeholder can decide the size of training data sometimes speed up
x_ph = tf.placeholder(tf.float32,[None,1])
y_ph = tf.placeholder(tf.float32,[None,1])

# design a hidden layer has 10 neurons(input to hidden1=1 to 10)
# input is one atrribution(only x) seen as one neuron = output
hidden_layer1 = add_layer(x_ph,1,10,activation_function=tf.nn.relu)
# design output layer
y_predict = add_layer(hidden_layer1,10,1,activation_function=None)

loss = tf.reduce_mean(tf.square(y_ph-y_predict))

optimizier = tf.train.GradientDescentOptimizer(0.2)
train_function = optimizier.minimize(loss)

init = tf.global_variables_initializer()





sess = tf.Session()
sess.run(init)

### for plot
fig = plt.figure()
axis = fig.add_subplot(1,1,1)
axis.scatter(x_real,y_real)
plt.ion() # turn the iteractive mode on
plt.show() #only plot once

for i in range(100):
    #sess.run(train_function)
    #if use placeholder remember to feed the data everytime use
    sess.run(train_function,feed_dict={x_ph:x_real,y_ph:y_real})
    if i % 10:
        try:
            axis.lines.remove(lines[0])
        except Exception:
            pass
        #print(sess.run(loss,feed_dict={x_ph:x_real,y_ph:y_real}))
        prediction_value = sess.run(y_predict,feed_dict={x_ph:x_real,y_ph:y_real})
        lines = axis.plot(x_real,prediction_value,'r-',lw=5)
        plt.pause(1)


