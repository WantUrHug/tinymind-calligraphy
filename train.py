import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
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

	sorted_logits_index = tf.argsort(logits, direction = "DESCENDING")
	threshold = tf.cast(tf.ones_like(labels)*5, tf.int32)#因为是top5，所以最大的5五个数值的序号为0、1、2、3、4，所以定为5
	top5 = tf.cast(tf.less(sorted_logits_index, threshold), tf.float32)#top5对应的数值为1，其他为0
	
	result = tf.reduce_sum(top5*labels, 1)

	return tf.reduce_mean(result)

if __name__ == "__main__":

	#设置训练时的常数
	TRAIN_BATCH_SIZE = 32
	TEST_BATCH_SIZE	= 128
	CLASS = 100
	IMG_H, IMG_W = 256, 256
	LEARNING_RATE = 10E-3
	TRAIN_STEPS = 100
	#多少步打印一次训练情况
	CHECK_STEP = 20
	#多少步保存一次数据
	SAVE_STEP = 200

	#准备数据
	train_dir = "D:\\BaiduNetdiskDownload\\TMD\\train\\train"
	test_dir = "D:\\BaiduNetdiskDownload\\TMD\\test1"
	test_csv = "D:\\BaiduNetdiskDownload\\TMD\\label-test1-fake.csv"
	dataset = MyDataset(train_dir, test_dir, test_csv)

	train_dataset = tf.data.Dataset.from_generator(dataset.train_datagen, output_types = (tf.float32, tf.float32))
	train_dataset = train_dataset.shuffle(20).batch(TRAIN_BATCH_SIZE).repeat()
	train_iterator = train_dataset.make_one_shot_iterator()
	next_train_data, next_train_labels = train_iterator.get_next()

	test_dataset = tf.data.Dataset.from_generator(dataset.test_datagen, output_types = (tf.float32, tf.float32))
	test_dataset = test_dataset.shuffle(20).batch(TEST_BATCH_SIZE).repeat()
	test_iterator = test_dataset.make_one_shot_iterator()
	next_test_data, next_test_labels = test_iterator.get_next()

	X = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, 1], name = "X")
	Y = tf.placeholder(tf.float32, [None, CLASS], name = "Y")

	net = MySimpleNet(CLASS)
	#前向传播
	outputs = net(X)
	#获取这个运算
	tf.add_to_collection("outputs", outputs)

	loss_op = tf.losses.softmax_cross_entropy(logits = outputs, onehot_labels = Y)
	train_op = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE).minimize(loss_op)
	top1_acc_op = CAL_TOP1_ACC(logits = outputs, labels = Y)
	top5_acc_op = CAL_TOP5_ACC(logits = outputs, labels = Y)

	history = {}
	history["train_loss"] = []
	history["train_top1_acc"] = []
	history["train_top5_acc"] = [] 
	history["test_loss"] = []
	history["test_top1_acc"] = []
	history["test_top5_acc"] = []

	config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
	#开始训练
	with tf.Session(config = config) as sess:		

		saver = tf.train.Saver(max_to_keep = 10)

		#初始化
		sess.run(tf.global_variables_initializer())

		for step in range(1, TRAIN_STEPS + 1):

			#获取训练集中的batch数据
			train_data, train_labels = sess.run((next_train_data, next_train_labels))
			if step % CHECK_STEP == 0:

				#获取测试集中的batch数据
				test_data, test_labels = sess.run((next_test_data, next_test_labels))
				#使用训练集数据更新模型，同时计算对应的损失和准确率
				_, train_loss, top1_acc, top5_acc = sess.run((train_op, loss_op, top1_acc_op, top5_acc_op), feed_dict = {X:train_data, Y:train_labels})
				#计算在测试集上的损失数值和top5准确率
				test_loss, test_top5_acc = sess.run((loss_op, top5_acc_op), feed_dict = {X:test_data, Y:test_labels})
				string_tuple = (step, train_loss, top1_acc*100, top5_acc*100, test_loss, test_top5_acc*100)
				print("Step %s, train loss %.2f, top1 acc %.2f%%, top5 acc %.2f%%, test loss %.2f, test top5 acc %.2f%%."%string_tuple)
			else:
				#只训练，不计算，提高效率
				sess.run(train_op, feed_dict = {X:train_data, Y:train_labels})

			#如果适时需要保存模型
			#if step % CHECK_STEP == 0:
			#	saver.save(sess, os.path.join("D:\\GitFile\\calligraphy\\models\\TrainResult", "simple"), global_step = step)

		num_check = range(1, len(history["train_loss"]) + 1)
		plt.subplot(1, 3, 1)
		plt.plot(num_check, history["train_loss"], "r", label = "train loss")
		plt.plot(num_check, history["test_loss"], 'b', label = "test loss")

		plt.subplot(1, 3, 2)
		plt.plot(num_check, history["train_top1_acc"], "r", label = "train top1 acc")

		plt.subplot(1, 3, 3)
		plt.plot(num_check, history["train_top5_acc"], "r", label = "train top5 acc")
		plt.plot(num_check, history["test_top5_acc"], "b", label = "test top5 acc")

		plt.legend()
		plt.show()