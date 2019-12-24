import numpy as np
import os
from PIL import Image

def get_largest_size_of_all(src_dir):
	'''
	因为书法图片，最好保持原本的形状，最好不要resize
	所以需要提前知道在所有训练图片中最大的尺寸是什么.
	Parameters:
		src_dir，目标文件夹
	'''
	max_h, max_w = 0, 0
	which_one = None
	for item in os.listdir(src_dir):
		item_path = os.path.join(src_dir, item)
		#print(item)
		for filename in os.listdir(item_path):
			filepath = os.path.join(item_path, filename)
			im = np.array(Image.open(filepath))
			im_h, im_w = im.shape[:2]
			if im_h > 1000 or im_w > 1000:
				continue
			if im_h > max_h:
				max_h = im_h
				which_one = filepath
			if im_w > max_w:
				max_w = im_w
				which_one = filepath

	print("最大的高度H是%d，最大的宽度W是%d"%(max_h, max_w))
	print("which one is %s"%which_one)

if __name__ == "__main__":

	train_dir = "D:\\BaiduNetdiskDownload\\TMD\\train\\train"
	get_largest_size_of_all(train_dir)