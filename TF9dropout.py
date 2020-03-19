'''
overfitting:when using the whole training dataset learns well for training but poor for test
result:training dataset performs better than test dataset
solution:use dropout can train with part of training set
need to add in the train step(training_fun) only remains xx% of prediction
this time 2 layers hidden layer and output layer both use add-layer
input: image data 0-9 size 8x8=64
tf.nn.dropout(x,keep_prob): With probability keep_prob, 
outputs the input element scaled up by 1 / keep_prob, 
otherwise outputs 0, so that the expected sum is unchanged.
ex:(x,0.6), 40% of data = 0 and 60% becomes 1/60%
why hidden layer with 50 neurons is better that 100? 
too many neurons can cause discrete?

'''
import tensorflow as tf 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data
digits = load_digits()
X = digits.data 
y = digits.target 
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3)

### hidden layer function
def add_layer(inputs,in_size,out_size,layer_name,activation_function=None):
    # use normal distribution function to generate a 
    #  random variables matrix better than zeros
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    # bias like a list recommend not 0
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    learning_fun = tf.matmul(tf.cast(inputs,tf.float32),Weights) + biases
    # add dropout this is where it works
    learning_fun = tf.nn.dropout(learning_fun,keep_prob)
    if activation_function is None:
        outputs = learning_fun
    else:
        outputs = activation_function(learning_fun)
    #tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs
###

### design input layer
# input is a 8x8 picture?, output is 10 number represent 0-9
keep_prob = tf.placeholder(tf.float32)
x_ph =  tf.placeholder(tf.float32,[None,64])
y_ph =  tf.placeholder(tf.float32,[None,10])
###

### design hidden layer 1
# output size =80 to see the problem of overfitting
hidden_layer = add_layer(x_ph,64,50,'hidden_layer1',tf.nn.tanh)
###

### design output layer(this time use add layer fun)
y_predict = add_layer(hidden_layer,50,10,'output_layer',tf.nn.softmax)
###

# calculate loss and use tensorbard to show
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ph*tf.log(y_predict),
                reduction_indices=[1]))
tf.summary.scalar('loss',cross_entropy)
# optimizer 
train_function = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# initialize all variables
init = tf.global_variables_initializer()

# define session
sess = tf.Session()
sess.run(init)
# add merged to use tensorboard
merged = tf.summary.merge_all()
# write and save the summary in tensorboard
writer_train = tf.summary.FileWriter("logs/train",sess.graph)
writer_test = tf.summary.FileWriter("logs/test",sess.graph)
for i in range(500):
    sess.run(train_function,feed_dict={x_ph:X_train,y_ph:y_train,keep_prob:0.5})
    if i % 50 == 0:
        result_train = sess.run(merged,feed_dict={x_ph:X_train,y_ph:y_train,keep_prob:1})
        result_test = sess.run(merged,feed_dict={x_ph:X_test,y_ph:y_test,keep_prob:1})
        writer_train.add_summary(result_train,i)
        writer_test.add_summary(result_test,i)