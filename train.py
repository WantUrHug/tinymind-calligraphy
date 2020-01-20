import tensorflow as tf
import numpy as np
from input_data import MyDataset
from models.simple import MySimpleNet

def CAL_TOP1_ACC(logits, labels):
	'''
	如果计算得到的概率最大的那个类别，和真实的类别是对应的，则说明是正确的。
	'''
	equality = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))

	return tf.reduce_mean(tf.cast(equality, tf.float32))

def CAL_TOP5_ACC(logits, labels):
	'''
	计算训练过程中的 TOP5 准确率，logits 是模型的输出，labels 是唯一的结果，比较输出中
	概率最高的前五个字符，如果包含了labels中指代的字符，即视为正确。
	'''

	#labels = tf.argmax(labels, 1)
	#batch_size = labels.shape[0]

	sorted_logits_index = tf.argsort(logits, direction = "DESCENDING")
	threshold = tf.cast(tf.ones_like(labels)*5, tf.int32)#因为是top5，所以最大的5五个数值的序号为0、1、2、3、4，所以定为5
	top5 = tf.cast(tf.less(sorted_logits_index, threshold), tf.float32)#top5对应的数值为1，其他为0
	
	result = tf.reduce_sum(top5*labels, 1)

	return tf.reduce_mean(result)

if __name__ == "__main__":

	#设置训练时的常数
	BATCH_SIZE = 32
	CLASS = 100
	IMG_H, IMG_W = 256, 256
	LEARNING_RATE = 10E-3
	TRAIN_STEPS = 1000

	train_dir = "D:\\BaiduNetdiskDownload\\TMD\\train\\train"
	dataset = MyDataset(train_dir)

	train_dataset = tf.data.Dataset.from_generator(dataset.train_datagen, output_types = (tf.float32, tf.float32))
	train_dataset = train_dataset.shuffle(20).batch(BATCH_SIZE).repeat()
	train_iterator = train_dataset.make_one_shot_iterator()

	next_train_data, next_train_labels = train_iterator.get_next()

	X = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, 1], name = "X")
	Y = tf.placeholder(tf.float32, [None, CLASS], name = "Y")

	net = MySimpleNet(CLASS)
	outputs = net(X)
	tf.add_to_collection("outputs", outputs)

	loss_op = tf.losses.softmax_cross_entropy(logits = outputs, onehot_labels = Y)
	train_op = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE).minimize(loss_op)
	top1_acc_op = CAL_TOP1_ACC(logits = outputs, labels = Y)
	top5_acc_op = CAL_TOP5_ACC(logits = outputs, labels = Y)

	history = {}
	history["train_loss"] = []
	history["train_acc"] = []

	config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
	with tf.Session(config = config) as sess:		

		sess.run(tf.global_variables_initializer())

		for step in range(1, TRAIN_STEPS + 1):

			train_data, train_labels = sess.run((next_train_data, next_train_labels))
			#train_labels = tf.one_hot(train_labels, depth = CLASS)
			#train_loss, top1_acc, top5_acc, _ = sess.run((loss_op, top1_acc_op, top5_acc_op, train_op), feed_dict = {X:train_data, Y:train_labels})
			_, train_loss, top1_acc, top5_acc= sess.run((train_op, loss_op, top1_acc_op, top5_acc_op), feed_dict = {X:train_data, Y:train_labels})
			#print("Step %s, train loss %.4f, top1 acc %.2f%, top5 acc %.2f%"%(step, train_loss, top1_acc*100, top5_acc*100))
			if step % 50 == 0:
				print("Step %s, train loss %.2f, top1 acc %.2f%%, top5 acc %.2f%%."%(step, train_loss, top1_acc*100, top5_acc*100))

