'''
use MNIST data
x: 28x28 matrix so size=784
y: 10x1 matrix, presents 0 to 9 if predicts one number related bit=1
softmax is the popular activation function
cross entropy to calculate loss
seperate dataset: training and test
'''
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
# read the data first and later will be used directly
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

# define placeholder for input data
x_ph = tf.placeholder(tf.float32,[None,784]) #None is how many samples,784 is one sample size
y_ph = tf.placeholder(tf.float32,[None,10])

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

### define compute_accuracy
def compute_accuracy(x_validation,y_validation):
    global y_predict
    y_interpredict = sess.run(y_predict,feed_dict={x_ph:x_validation})
    # tf.argmax() index of max value in certain dimension,0 is line 1 is column
    # tf.equal returns a tensor of type bool
    correct_prediction = tf.equal(tf.argmax(y_interpredict,1),tf.argmax(y_validation,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={x_ph:x_validation,y_ph:y_validation})
    return result
###

### define output layer
y_predict = add_layer(x_ph,784,10,activation_function=tf.nn.softmax)
###

# loss
cross_entroy = tf.reduce_mean(-tf.reduce_sum(y_ph*tf.log(y_predict),
                reduction_indices=[1]))

# training function
train_function = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entroy)

# initialize all variables
init = tf.global_variables_initializer()

# define session
sess = tf.Session()
sess.run(init)

for i in range(1000):
    # mnist dataset is big so use batch to speed up training
    batch_x_input,batch_y_input = mnist.train.next_batch(100)
    sess.run(train_function,feed_dict={x_ph:batch_x_input,y_ph:batch_y_input})
    if i % 50 ==0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))
