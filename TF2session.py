import tensorflow as tf 

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])

product = tf.matmul(matrix1,matrix2) # =np.dot(m1,m2)

# method 1
sess = tf.Session()
result1 = sess.run(product)
print(result1)
sess.close()

# method 2 don't need to use close
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)
