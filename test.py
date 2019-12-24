import tensorflow as tf

tf.enable_eager_execution()

a = tf.Variable([[1,1],[2,2]])
b = tf.Variable([[1,3],[5,6]])
print(a+b)