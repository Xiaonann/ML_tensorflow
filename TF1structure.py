'''
tensorflow structure
'''
import tensorflow as tf
import numpy as np 

# learned function is y = 0.1 * x + 0.3,x is random variables
x_input = np.random.rand(100).astype(np.float)
y_input = x_input * 0.1+0.3

### create tensorflow structure capital menas matrix
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0)) #(dimension,start,stop)
biases = tf.Variable(tf.zeros([1])) #initialize bias = 1
y_predict = Weights*x_input + biases
# loss function
loss = tf.reduce_mean(tf.square(y_predict-y_input))
# optimal function
optimizer = tf.train.GradientDescentOptimizer(0.5) # learning rate = 0.5
train = optimizer.minimize(loss)
# only with this step can start
init = tf.initialize_all_variables() 
### end tensorflow structure


sess = tf.Session() 
#point out the start point
sess.run(init) 

for step in range(100):
    sess.run(train)
    if step % 10 == 0:
        print(step,sess.run(Weights),sess.run(biases))