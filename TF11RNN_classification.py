'''
one cell (like filter in CNN) used to operate one sequence
for vanilla recurrent neural networks and GRU’s, the output = hidden state
LSTM 
read image by row and to identify the number
need to change diemension for cal
'''
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

# hyperparameters
learning_rate = 0.001
train_iteration = 100000
batch_size = 128

# MNIST is 28*28
n_input = 28 # input(one row) size is 28
n_step = 28 # each time step read one row
n_neurons = 128 # num of neurons in each hidden layer
n_output = 10 # output 0-9



# define placeholder for input data
x_ph = tf.placeholder(tf.float32,[None,n_input,n_step]) #None is how many samples,784 is one sample size
y_ph = tf.placeholder(tf.float32,[None,n_output])

# define weights and bias for adding hidden layers before and after RNN unit
weights = {'in':tf.Variable(tf.random_normal([n_input,n_neurons])),
'out':tf.Variable(tf.random_normal([n_neurons,n_output]))}
biases = {'in':tf.Variable(tf.constant(0.1,shape=[n_neurons,])),
'out':tf.Variable(tf.constant(0.1,shape=[n_output,]))}


### build network
## hidden layer after input
def RNN(X,weights,biases):
    ## one hidden layer after input
    # X(128 batch, 28 step, 28 input) for cal need to reshape
    # ->>(128*28, 28)
    X = tf.reshape(X,[-1,n_input])
    hidden_in = tf.matmul(X,weights['in']) + biases['in'] #(128*28,n_neurons)
    hidden_in = tf.reshape(hidden_in,[-1,n_step,n_neurons]) #(128,28 step,n_neurons)
    
    ## cell
    # Initialize the basic LSTM cell.
    # state_is_tuple: True, states are 2-tuples of the c_state and m_state
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_neurons,forget_bias=1.0,state_is_tuple=True)
    lstm_ini_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)
    # input to `cell` at each time step [batch_size, ...]
    # time_major If true, these Tensors must be shaped [max_time, batch_size, depth] 
    #            If false, these Tensors must be shaped [batch_size, max_time, depth]
    # states[0] = c_state = (current cell,hidden state) hidden state is computed by cell and output gate
    # state[1] = m_state = hidden state = outputs
    # dimension: outputs(128 batch, 28 step, 128 neuron), states[0] = states[1] (128 batch, n_neuron)
    outputs,states = tf.nn.dynamic_rnn(lstm_cell,hidden_in,initial_state=lstm_ini_state,time_major=False)
 
    # hidden layer before output
    hidden_out = tf.matmul(states[1],weights['out']) + biases['out'] #(128*28,n_output)
    # another way to cal, use first and third column of outputs 
    # first transpose get(28 step, 128 batch, 128 neuron) then get list step*[(batch,neuron)]
    #outputs = tf.unpack(tf.transpose(outputs,[1,0,2])) 
    #hidden_out = tf.matmul(outputs[-1],weights['out']) + biases['out']
    result = hidden_out
    return result

y_prediction = RNN(x_ph,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_ph, logits = y_prediction))
train_function = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# cal accuracy
correct_prediction = tf.equal(tf.argmax(y_prediction,1),tf.argmax(y_ph,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# initialize all variables
init = tf.global_variables_initializer()

# define session
sess = tf.Session()
sess.run(init)
step = 0
while step*batch_size<train_iteration:
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = batch_x.reshape([batch_size,n_step,n_input])
    sess.run(train_function,feed_dict={x_ph:batch_x,y_ph:batch_y})
    if step % 20 == 0:
        print(sess.run(accuracy,feed_dict={x_ph:batch_x,y_ph:batch_y}))
        step += 1









