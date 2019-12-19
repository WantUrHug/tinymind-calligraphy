import tensorflow as tf
import numpy as np
from tensorflow.layers import conv2d, max_pooling2d, dense, batch_normalization, flatten
from tensorflow.nn import relu

class MySimpleNet():

	def __init__(self, output_channels):

		self.output_channels = output_channels

	def __call__(self, input):

		return self.inference(input)

	def inference(self, input, istraining = True):

		output = conv2d(input, 16, kernel_size = (3, 3), strides = (1, 1), padding = "SAME", activation = None)
		output = batch_normalization(input, istraining)
		output = relu(output)

		output = conv2d(input, 32, kernel_size = (3, 3), strides = (1, 1), padding = "SAME", activation = None)
		output = batch_normalization(input, istraining)
		output = relu(output)

		output = conv2d(input, 64, kernel_size = (3, 3), strides = (1, 1), padding = "SAME", activation = None)
		output = batch_normalization(input, istraining)
		output = relu(output)

		output = flatten(output)

		output = dense(output, 4096)
		output = dense(output, 1024)
		output = dense(output, self.output_channels)

		return output

if __name__ == "__main__":

	net = MySimpleNet(10)
	a = tf.placeholder(tf.float32, [None, 512, 512, 3])
	print(net(a))