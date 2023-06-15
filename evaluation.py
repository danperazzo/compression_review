import cv2
import os
from metrics import lpips, PSNR, ssim
import time
from tqdm import tqdm
import pandas as pd

def imwrite(results_folder, name ,img):
    cv2.imwrite(results_folder +'/' +name,img)

def imread(data_folder, name):
    img_rgb = cv2.imread(data_folder + '/' + name)
    return img_rgb

def prepare_results_folder(order, block_shape,  method):
    results_folder = 'results_' + f'order_{order:.3f}_'+f'blocks_{block_shape}'
    
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    method_folder = results_folder+'/'+method
    if not os.path.exists(method_folder):
        os.mkdir(method_folder)

    return method_folder

def average_l(lst):
    return sum(lst) / len(lst)


def test_for_parameters(compressor, data_folder, results_method_folder):

	img_list = os.listdir( data_folder )
	ssim_list = []
	psnr_list = []
	lpips_list = []
	time_encode = []
	time_decode = []

	for imname in tqdm(img_list):

		img_rgb = imread(data_folder, imname)
		start_encode = time.time()
		img_rgb_code = compressor.encode_rgb(img_rgb)
		end_encode = time.time()

		start_decode = time.time()
		img_rgb_compress = compressor.decode_rgb(img_rgb_code)
		end_decode = time.time()

		imwrite(results_method_folder, imname, img_rgb_compress)

		ssim_val = ssim(img_rgb, img_rgb_compress)
		psnr_val = PSNR(img_rgb, img_rgb_compress)
		lpips_val = lpips(img_rgb, img_rgb_compress)

		time_encode_val = end_encode - start_encode
		time_decode_val = end_decode - start_decode


		ssim_list.append(ssim_val)
		psnr_list.append(psnr_val)
		lpips_list.append(lpips_val)
		time_encode.append(time_encode_val)
		time_decode.append(time_decode_val)


	ssim_avg = average_l(ssim_list)
	psnr_avg = average_l(psnr_list)
	lpips_avg = average_l(lpips_list)
	time_encode_avg = average_l(time_encode)
	time_decode_avg = average_l(time_decode)


	df = pd.DataFrame({'image_name': (img_list+['mean']),
					'ssim': ssim_list + [ssim_avg],
					'psnr': psnr_list + [psnr_avg],
					'lpips': lpips_list + [lpips_avg],
					'time_encode':time_encode + [time_encode_avg],
					'time_decode':time_decode + [time_decode_avg]})

	csv_path = results_method_folder +'/results_num.csv'
	df.to_csv(csv_path)  