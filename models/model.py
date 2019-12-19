import tensorflow as tf
import numpy as np
from tensorflow.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization
from tensorflow.nn import relu

def conv2d_bn(x, nb_filter, kernel_size, strides = (1, 1), padding = "SAME"):
	"""
	conv2d -> batch normalization -> relu.
	"""
	x = Conv2D(
		nb_filter,
		kernel_size,
		strides,
		padding,
		kernel_regularizer = tf.contrib.layers.l1_regularizer(0.001))(x)
	x = BatchNormalization()(x)
	x = relu(x)

	return x

def shortcut(input, residual):
	"""
	shortcut连接.
	"""

	input_shape = input.shape
	residual_shape = residual.shape
	#通常卷积越深，特征是越小，所以如果shortcut前后的尺寸不一(通常要的)，就需要加上一个卷积变换
	#例如input_shape = (N, 40, 40, 64), residual_shape = (N, 20, 20, 128)
	#步长如下
	#如何理解呢?因为我们需要缩放，那么要不就池化要不就卷积来有效的缩小，这里选择的是卷积，设置好合适的步长
	stride_height = int(input_shape[1]/residual_shape[1])
	stride_width = int(input_shape[2]/residual_shape[2])

	equal_channels = (input_shape[3] == residual_shape[3])

	identity = input

	if stride_height > 1 or stride_width > 1 or not equal_channels:
		identity = Conv2D(
			residual_shape[3],
			kernel_size = (1,1),
			strides = (stride_height, stride_width),
			padding = "VALID",
			kernel_regularizer = tf.contrib.layers.l1_regularizer(0.001))(input)

	#把两个维度大小相同的向量相加，矩阵加法
	return identity + residual

def basic_block(nb_filter, strides = (1,1)):

	def f(input):

		conv1 = conv2d_bn(input, nb_filter, kernel_size = (3,3), strides = strides)
		residual = conv2d_bn(conv1, nb_filter, kernel_size=(3, 3))

		return shortcut(input, residual)

	return f

if __name__ == "__main__":

	val = np.ones([5, 160, 160, 5])
	x = tf.Variable(val, dtype = tf.float32)
	y = conv2d_bn(x, 10, 3)
	print(y)