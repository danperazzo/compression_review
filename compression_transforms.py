import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import fftpack

all_methods = ['fourrier', 'dct', 'svd', 'kl']


def dct2(f):
    """
    Discrete Cosine Transform in 2D.
    """
    return np.transpose(fftpack.dct(
           np.transpose(fftpack.dct(f, norm = "ortho")), norm = "ortho"))

def idct2(f):
    """
    Inverse Discrete Cosine Transform in 2D.
    """
    return np.transpose(fftpack.idct(
           np.transpose(fftpack.idct(f, norm = "ortho")), norm = "ortho"))




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

def kl_compression_block(image,order):
	size = image.shape[1]

	# Make columns of image 0 mean
	mean = np.mean(image,axis=0) 
	image = image - mean

	# Decorrelate columns of image
	covariance = np.cov(image, rowvar=False)
	_, eig = np.linalg.eigh(covariance) # Eigenvectors ordered low-to-high 
	KL = image @ eig

	# Quantization of the transformed image
	KL_compressed = KL[:,size-order:size]
	eig_compressed = eig[:,size-order:size] 

	#Result image
	image_comp = (KL_compressed @ np.transpose(eig_compressed)) + mean 
	return image_comp

def svd_compression_block(image,order):
	[U,S,V] = np.linalg.svd(image, full_matrices=False, compute_uv=True, hermitian=False)
	S_compressed = S[:order]
	U_compressed = U[:,:order]
	V_compressed = V[:order,:]
	img_compressed = np.dot(U_compressed * S_compressed, V_compressed)
	return np.round(img_compressed)

def fourrier_compression_block(image, order):
	
	Atlow = np.fft.fft2(image)
	Atlow[order:,order:] = 0
	
	image_compressed = np.fft.ifft2(Atlow).real

	return image_compressed

def dct_compression_block(image, order):	
	Atlow = dct2(image)
	Atlow[order:,order:] = 0
	
	image_compressed = idct2(Atlow)

	return image_compressed

def kl_compressor(image_blocks, image,block, block_size, order):
	image_blocks_compressed = []
	for image_col in image_blocks:

		image_block = np.reshape(image_col,block)
		kl_compression_image = kl_compression_block(image_block,order)
		kl_compression_col = np.reshape(kl_compression_image,block[0]*block[1])
		image_blocks_compressed.append(kl_compression_col)

	image_compressed = np.stack(image_blocks_compressed)
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

def dct_compression(image_blocks, image,block, block_size,order):
	image_blocks_compressed = []
	for image_col in image_blocks:
		
		image_block = np.reshape(image_col,block)
		compression_image = dct_compression_block(image_block,order)
		compression_col = np.reshape(compression_image,block[0]*block[1])
		image_blocks_compressed.append(compression_col)

	
	image_compressed = np.stack(image_blocks_compressed)

	return image_compressed

def block_compressor(image, order = 5, block = (8,8), compression='kl'):


	if compression == 'kl':
		image = image.astype(np.double)
		
		block_size = block[0] * block[1]
		image_blocks = im2col(image,block)

		image_col = kl_compressor(image_blocks,image, block,block_size, order)
		image_comp = col2im(image_col, (image.shape[0],image.shape[1]), block)

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

	elif compression == 'dct':
		image = image.astype(np.double)

		block_size = block[0] * block[1]
		image_blocks = im2col(image,block)

		image_col = dct_compression(image_blocks,image, block,block_size,order)
		image_comp = col2im(image_col, (image.shape[0],image.shape[1]), block)


	return image_comp


def compress_rgb(img_rgb, order = 5, block = (8,8), compression = 'fourrier'):

    img_c_stack = []
    for i in range(3):
            img_c = img_rgb[:,:,i] 
            img_c_compressed = block_compressor(img_c,order,block = block, compression = compression)
            img_c_stack.append(img_c_compressed)

    img_rgb = np.dstack(img_c_stack)
    img_rgb = np.around(img_rgb).astype(int)

    return img_rgb