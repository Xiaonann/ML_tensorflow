import tensorflow as tf 

# variable
var = tf.Variable(1, name='firstvariable')
#print(var.name)

# constant

con = tf.constant(1)

sum_varcon = tf.add(var,con)
# update var by adding con on it
updates  = tf.assign(var,sum_varcon) 

# !!! must have this when considering variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(5):
        sess.run(updates)
        # must use sess.run() to get any value in the session
        print(sess.run(var))
