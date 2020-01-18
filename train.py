import tensorflow as tf
import numpy as np
from input_data import MyDataset
from models.simple import MySimpleNet

def CAL_TOP1_ACC(logits, labels):
	

if __name__ == "__main__":

	BATCH_SIZE = 32
	CLASS = 100
	IMG_H, IMG_W = 256, 256
	LEARNING_RATE = 10E-4
	TRAIN_STEPS = 100

	train_dir = "D:\\BaiduNetdiskDownload\\TMD\\train\\train"
	dataset = MyDataset(train_dir)

	train_dataset = tf.data.Dataset.from_generator(dataset.train_datagen, output_types = (tf.float32, tf.int16))
	train_dataset = train_dataset.shuffle(20).batch(BATCH_SIZE).repeat()
	train_iterator = train_dataset.make_one_shot_iterator()

	next_train_data, next_train_labels = train_iterator.get_next()

	X = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, 1], name = "X")
	Y = tf.placeholder(tf.int16, [None, CLASS], name = "Y")

	outputs = MySimpleNet[X]
	tf.add_to_collection("outputs", outputs)

	loss_op = tf.losses.softmax_cross_entropy(logits = outputs, labels = Y)
	train_op = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE).minimize(loss_op)

	#top1_acc
	#top5_acc

	history = {}
	history["train_loss"] = []
	history["train_acc"] = []

	config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
	with tf.Session(config = config) as sess:

		sess.run(tf.global_variables_initializer())

		for step in range(1, TRAIN_STEPS + 1):

			train_data, train_labels = sess.run((next_train_data, next_train_labels))
			train_loss, _ = sess.run((loss_op, train_op), feed_dict = {X:train_data, Y:train_labels})


