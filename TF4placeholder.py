'''
tf.placeholder(dtype,shape,name) dtypemust be given shape
can be written by[number of samples,size of one sample]
'''
import tensorflow as tf 

# only a symbol without value so when use it should be fed value
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)

# use feed_dict{} to feed placeholder it is a dictory key:value
with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[1],input2:[3]}))

