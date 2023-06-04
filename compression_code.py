import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from numpy.fft import fft2, ifft2, fftshift, ifftshift


def im2col(image,block): # to divide the image into blocks
		image_block = []
		block_height = block[1]
		block_width = block[0]
		block_size = block_height * block_width
		for j in range(0, image.shape[1], block_height):
				for i in range(0, image.shape[0], block_width):
						image_block.append(np.reshape(image[i:i+block_width, j:j+block_height], block_size))
		image_block = np.asarray(image_block).astype(float)
		return image_block

def col2im(mtx, image_size, block): # to combine the blocks back into image
		p, q = block
		sx = image_size[1]
		sy = image_size[0]
		result = np.zeros(image_size)  
		col = 0
		for j in range(0,sx,q):
				 for i in range(0,sy,p):
						 result[i:i+q, j:j+p] = mtx[col].reshape((block))
						 col += 1
		return result

def kl_compressor(image_blocks, image,block, block_size, order):

	mean = np.mean(image_blocks,0) # calculate the mean of the block

	image_centered = np.transpose(image_blocks) - mean.reshape((block_size,1)) # make it zero mean
	covariance = np.cov(image_centered) # find the covariance matrix

	eig_val, eig_vec = np.linalg.eig(np.transpose(covariance)) # Finding eigen vectors of covariance matrix
	idx = eig_val.argsort()[::-1] # Sort the eigen vector matrix from highest to lowest
	eig_val_sorted = eig_val[idx]
	eig_vec_sorted = eig_vec[:,idx]
	y = np.matmul(np.transpose(eig_vec_sorted),image_centered)

	y[order:block_size,:] = np.zeros((block_size - order,y.shape[1])); # make the last block_size-n eigen vectors zero.
	z2 = np.matmul(np.linalg.inv(np.transpose(eig_vec_sorted)),y); # For restoring the image from 
	x2 = z2 + mean.reshape((block_size,1)); # Add the mean for plotting

	image_comp = col2im(np.transpose(x2), (image.shape[0],image.shape[1]), block) # compressed image

	return image_comp

def svd_compression_block(image,order):
	[U,S,V] = np.linalg.svd(image, full_matrices=False, compute_uv=True, hermitian=False)
	S_compressed = S[:order]
	U_compressed = U[:,:order]
	V_compressed = V[:order,:]
	img_compressed = np.dot(U_compressed * S_compressed, V_compressed)
	return np.round(img_compressed)

def fourrier_compression_block(image, order):
	perc = order/image.shape[0]
	
	Bt = np.fft.fft2(image)
	Btsort = np.sort(np.abs(Bt.reshape(-1)))

	thresh_ind = int((1-perc)*len(Btsort))
	thresh = Btsort[thresh_ind]
	ind = np.abs(Bt) > thresh
	Atlow =  Bt * ind
	
	image_compressed = np.fft.ifft2(Atlow).real

	return image_compressed

def svd_compression(image_blocks, image,block, block_size,order):

	image_blocks_compressed = []
	for image_col in image_blocks:
		
		image_block = np.reshape(image_col,block)
		svd_compression_image = svd_compression_block(image_block,order)
		svd_compression_col = np.reshape(svd_compression_image,block[0]*block[1])
		image_blocks_compressed.append(svd_compression_col)

	
	image_compressed = np.stack(image_blocks_compressed)

	return image_compressed

def fourrier_compression(image_blocks, image,block, block_size,order):
	image_blocks_compressed = []
	for image_col in image_blocks:
		
		image_block = np.reshape(image_col,block)
		compression_image = fourrier_compression_block(image_block,order)
		compression_col = np.reshape(compression_image,block[0]*block[1])
		image_blocks_compressed.append(compression_col)

	
	image_compressed = np.stack(image_blocks_compressed)

	return image_compressed

def block_compressor(image, order = 5, block = (16,16), compression = 'kl'):


	if compression == 'kl':
		image = image.astype(np.double)
		
		block_size = block[0] * block[1]
		image_blocks = im2col(image,block)

		image_comp = kl_compressor(image_blocks,image, block,block_size, order)

	elif compression == 'svd':
		image = image.astype(np.double)

		block_size = block[0] * block[1]
		image_blocks = im2col(image,block)

		image_col = svd_compression(image_blocks,image, block,block_size,order)
		image_comp = col2im(image_col, (image.shape[0],image.shape[1]), block)

	elif compression == 'fourrier':
		image = image.astype(np.double)

		block_size = block[0] * block[1]
		image_blocks = im2col(image,block)

		image_col = fourrier_compression(image_blocks,image, block,block_size,order)
		image_comp = col2im(image_col, (image.shape[0],image.shape[1]), block)


	return image_comp

order = 8

img_rgb = np.array(Image.open('datasets/kodak/kodim01.png'))

plt.imshow(img_rgb.astype(int))
plt.show()
 
#convert the image into grayscale

img_c_stack = []
for i in range(3):
		img_c = img_rgb[:,:,i] 
		img_c_compressed = block_compressor(img_c,order,block = (128,128), compression = 'fourrier')
		img_c_stack.append(img_c_compressed)


img_rgb = np.dstack(img_c_stack)
img_rgb = np.around(img_rgb).astype(int)

plt.imshow(img_rgb)
plt.show()

