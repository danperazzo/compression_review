import os

from compression_transforms import compression_factory
from utils import imwrite, imread, prepare_results_folder, PSNR, ssim, lpips, average_l
import pandas as pd
import time
from tqdm import tqdm



orders = [0.20, 0.30,  0.50]
all_methods = ['svd', 'dft', 'pca']
block_shapes = [8, 32, 64]
data_folder = 'datasets/kodak/'


def test_for_parameters(compressor, data_folder):

	results_folder = prepare_results_folder(compressor.order, compressor.block_size, all_methods)

	img_list = os.listdir( data_folder )
	ssim_list = []
	psnr_list = []
	lpips_list = []
	time_elapsed = []

	for imname in tqdm(img_list):

		img_rgb = imread(data_folder, imname)
		start = time.time()
		img_rgb_compress = compressor.compress_rgb(img_rgb)
		end = time.time()
		imwrite(results_folder , method ,imname,img_rgb_compress)

		ssim_val = ssim(img_rgb,img_rgb_compress)
		psnr_val = PSNR(img_rgb,img_rgb_compress)
		lpips_val = lpips(img_rgb, img_rgb_compress)
		time_val = end - start


		ssim_list.append(ssim_val)
		psnr_list.append(psnr_val)
		lpips_list.append(lpips_val)
		time_elapsed.append(time_val)


	ssim_avg = average_l(ssim_list)
	psnr_avg = average_l(psnr_list)
	lpips_avg = average_l(lpips_list)
	time_avg = average_l(time_elapsed)


	df = pd.DataFrame({'image_name': (img_list+['mean']),
					'ssim': ssim_list + [ssim_avg],
					'psnr': psnr_list + [psnr_avg],
					'lpips': lpips_list + [lpips_avg],
					'time':time_elapsed + [time_avg]})

	csv_path = results_folder + '/' + method +'/results_num.csv'
	df.to_csv(csv_path)  

for block in block_shapes:
	for order in orders:
		for method in all_methods:

			compressor = compression_factory(method, order, block)
			print(f'Order: {order}, method: {method}, block: {block}')
			test_for_parameters(compressor, data_folder)

