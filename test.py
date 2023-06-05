import os

from compression_transforms import compress_rgb, all_methods
from utils import imwrite, imread, prepare_results_folder

order = 1
method = 'kl'
block_shape = (8,8)
data_folder = 'datasets/kodak/'

results_folder = prepare_results_folder(order, block_shape, all_methods)

img_list = os.listdir( data_folder )

for imname in img_list:

	img_rgb = imread(data_folder, imname) 
	img_rgb_compress = compress_rgb(img_rgb, order,block_shape, compression = method)
	imwrite(results_folder , method ,imname,img_rgb_compress)

	print(imname)


print(img_rgb.shape)

