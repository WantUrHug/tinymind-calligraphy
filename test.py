import tensorflow as tf
import os
import numpy as np
from input_data import MyDataset

def get_top5_onehot(logits):
	'''
	获得 top5 的五个 index，仍是one-hot的形式，然后才方便去获取对应的字符
	'''
	sorted_logits_index = tf.argsort(logits, direction = "DESCENDING")
	threshold = tf.cast(tf.ones_like(sorted_logits_index)*5, tf.int32)
	top5 = tf.cast(tf.less(sorted_logits_index, threshold), tf.float32)

	return top5

def top5_to_index(top5_value):
	'''
	输入是一个 numpy.ndarray，其中是 0 和 1，需要提取 1 所在的 index，重新返回
	'''
	outputs = []
	for array in top5_value:
		ids = []
		for i, value in enumerate(array):
			if value == 1:
				ids.append(i)
		outputs.append(ids)
	return outputs


if __name__ == "__main__":
	
	final_dir = "D:\\BaiduNetdiskDownload\\TMD\\test2"
	train_dir = "D:\\BaiduNetdiskDownload\\TMD\\train\\train"
	dataset = MyDataset(train_dir = train_dir, final_dir = final_dir)
	#dataset = MyDataset()

	models_dir = "D:\\GitFile\\calligraphy\\models\\TrainResult"
	ckpt = tf.train.get_checkpoint_state(models_dir)
	#print(ckpt)
	config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
	with tf.Session(config = config) as sess:

		lastest_model = ckpt.model_checkpoint_path
		print(lastest_model)
		saver = tf.train.import_meta_graph(lastest_model + ".meta")
		saver.restore(sess, lastest_model)

		graph = tf.get_default_graph()

		outputs = tf.get_collection("outputs")[0]
		X = graph.get_tensor_by_name("X:0")
		#Y = graph.get_tensor_by_name("Y:0")

		top5 = get_top5_onehot(outputs)

		with open("result.csv",  "w", encoding = "utf-8") as file:

			file.writelines("filename, label\n")

			for i, (name_list, final_data) in enumerate(dataset.final_datagen()):

				top5_onehot = sess.run(top5, feed_dict = {X: final_data})
				word_ids = top5_to_index(top5_onehot)
				#print(word_ids)
				#for j, top5_value in enumerate(top5_onehot):
				for j, index in enumerate(word_ids):
					word_ls = [dataset.label_dict[ind] for ind in index]
					#print(word_ls)
					words = word_ls[0] + word_ls[1] + word_ls[2] + word_ls[3] + word_ls[4]
					file.writelines(name_list[j] + "," + words + "\n")
			
