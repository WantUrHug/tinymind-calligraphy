import tensorflow as tf
import os
from PIL import Image
import numpy as np

class Dataset():

	def __init__(self, train_dir, test_dir = None, img_h = 512, img_w = 512):

		self.train_dir = train_dir
		self.test_dir = test_dir

		self.train_word_list = list(os.listdir(self.train_dir))
		self.img_h, self.img_w = img_h, img_w
		#num_class = len(word_list)

		#num_per_class = len(os.path.join(base_dir, os.listdir(base)[0]))
		#NPC = num_per_class

	def train_datagen(self):

	#前面的class_num就是表示了，有多少个文件夹多少个类
	#我所返回的标签，也就是其中的序号即可
		for i, word in enumerate(self.train_word_list):
			#for word_index in range(num_class):
			word_path = os.path.join(self.train_dir, word)
			for img_name in os.listdir(word_path):
				print(os.path.join(word_path, img_name))
				im = Image.open(os.path.join(word_path, img_name))
			#都是灰度图
				if im == None:
					print("read error happen.")
					continue
			
				im = im.resize((self.img_h, self.img_w))

				yield np.array(im), i

if __name__ == "__main__":

	train_dir = "D:\\BaiduNetdiskDownload\\TMD\\train\\train"
	dataset = Dataset(train_dir)
	g = dataset.train_datagen()
	print(next(g))