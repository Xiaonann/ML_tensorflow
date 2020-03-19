'''
Filter size f_w*f_l*input channels, filter_num = feature_num
Formular is similar with classification y = W*x+b,just using convolution operation

Popular one: LeNET-5, Alexnet

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
keep_prob = tf.placeholder(tf.float32)
# reshape(tensor,shape), shape [batch,width,length,channel]
# If one component of shape is the special value -1, the size of that dimension 
# is computed(don't need to pre-defined) so that the total size remains constant.
x_image = tf.reshape(x_ph,[-1,28,28,1])


### define weight and bias variable functions
def weight_variable(shape):
    #tf.truncated_normal random values from a turncatted normol distributution
    # The generated values follow a normal distribution with specified 
    # mean and standard deviation, except that values whose magnitude 
    # is more than 2 standard deviations from the mean are dropped and re-picked
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
###

### define convolutional network
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pooling(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


### CNN 
## 1st convolutional layer
W_conv1 = weight_variable([5,5,1,32]) # 5x5 patch, input size 1, output size 32
b_conv1 = bias_variable([32])
# add a nonlinear activation(ReLU)
hidden_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
pooling_conv1 = max_pooling(hidden_conv1)

## 2nd convolutional layer
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
hidden_conv2 = tf.nn.relu(conv2d(pooling_conv1,W_conv2)+b_conv2)
pooling_conv2 = max_pooling(hidden_conv2)

## 1st fully connected layer
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
# [n_sample,7,7,64]->> [n_sample,7*7*64]
flat_fc1 = tf.reshape(pooling_conv2,[-1,7*7*64])
hidden_fc1 = tf.nn.relu(tf.matmul(flat_fc1,W_fc1) + b_fc1)
# aviod overfitting
hidden_fc1_drop = tf.nn.dropout(hidden_fc1,keep_prob)

## 2st fully connected layer
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_predict = tf.nn.softmax(tf.matmul(hidden_fc1_drop,W_fc2) + b_fc2)



### define compute_accuracy
def compute_accuracy(x_validation,y_validation):
    global y_predict
    y_interpredict = sess.run(y_predict,feed_dict={x_ph:x_validation,keep_prob:1})
    # tf.argmax() index of max value in certain dimension,0 is line 1 is column
    # tf.equal returns a tensor of type bool
    correct_prediction = tf.equal(tf.argmax(y_interpredict,1),tf.argmax(y_validation,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={x_ph:x_validation,y_ph:y_validation,keep_prob:1})
    return result
###


# loss
cross_entroy = tf.reduce_mean(-tf.reduce_sum(y_ph*tf.log(y_predict),
                reduction_indices=[1]))

# training function (complex NN use Adam rather than GradientDescent)
train_function = tf.train.AdamOptimizer(1e-4).minimize(cross_entroy)

# initialize all variables
init = tf.global_variables_initializer()

# define session
sess = tf.Session()
sess.run(init)

for i in range(1000):
    # mnist dataset is big so use batch to speed up training
    batch_x_input,batch_y_input = mnist.train.next_batch(200)
    sess.run(train_function,feed_dict={x_ph:batch_x_input,y_ph:batch_y_input,keep_prob:0.5})
    if i % 50 ==0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))


