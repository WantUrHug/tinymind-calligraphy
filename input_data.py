import tensorflow as tf
import os
from PIL import Image
import numpy as np

class MyDataset():

	def __init__(self, train_dir, test_dir = None, test_csv = None, img_h = 256, img_w = 256):

		self.train_dir = train_dir
		self.test_dir = test_dir
		self.test_csv = test_csv

		self.train_word_list = list(os.listdir(self.train_dir))
		#建立两个字典，一个是从数字索引到中文字符，另一个是从中文字符到数字索引
		self.label_dict = {}
		self.word_dict = {}
		for i, word in enumerate(self.train_word_list):

			self.label_dict[i] = word
			self.word_dict[word] = i

		self.img_h, self.img_w = img_h, img_w
		self.num_class = len(self.train_word_list)

		#num_per_class = len(os.path.join(base_dir, os.listdir(base)[0]))
		#NPC = num_per_class

	def train_datagen(self):

		#前面的class_num就是表示了，有多少个文件夹多少个类
		#我所返回的标签，也就是其中的序号即可
		for i, word in enumerate(self.train_word_list):

			#self.label_dict[i] = word
			#self.word_dict[word] = i

			#for word_index in range(num_class):
			word_path = os.path.join(self.train_dir, word)
			output_label = np.zeros([self.num_class])
			output_label[i] = 1
			for img_name in os.listdir(word_path):
				#print(os.path.join(word_path, img_name))
				img_path = os.path.join(word_path, img_name)
				#都是灰度图

				im = self.preprocess(img_path, (self.img_h, self.img_w))

				yield im, output_label

	def test_datagen(self):
		
		vec = np.loadtxt(self.test_csv, dtype = str, delimiter = ",", skiprows = 1, encoding = "utf-8")
		#print(vec)

		for img_info in vec:
			img_path = os.path.join(self.test_dir, img_info[0])
			im = self.preprocess(img_path, (self.img_h, self.img_w))
			output_label = np.zeros([self.num_class])
			label_index = [self.word_dict[i] for i in img_info[1]]
			output_label[label_index] = np.ones([5])

			yield im, output_label

	def preprocess(self, img_path, size = (256, 256), istraining = False):

		im = Image.open(img_path)
			#都是灰度图
		if im:
			pass
		else:
			print("read error happen.")
			return -1
			
		im = im.resize(size)
		im = (np.array(im).reshape((size[0], size[1], 1))/255.0-0.5)*2

		return im

		'''
		padding_h, padding_w = padding_size
		im_h, im_w = im.shape[:2]
		if im_h < padding_h and im_w < padding_w:
			#输入图片尺寸整体小于既定尺寸，无需缩放直接全1填充
			output = np.zeros(padding_size)
			output[:im_h, :im_w] = im

		#缩放到-1到1
		im = (im/255.0 - 0.5)*2
		'''

if __name__ == "__main__":

	train_dir = "D:\\BaiduNetdiskDownload\\TMD\\train\\train"
	test_dir = "D:\\BaiduNetdiskDownload\\TMD\\test1"
	test_csv = "D:\\BaiduNetdiskDownload\\TMD\\label-test1-fake.csv"
	dataset = MyDataset(train_dir, test_dir, test_csv)
	g = dataset.train_datagen()
	g2 = dataset.test_datagen()
	print(next(g2))