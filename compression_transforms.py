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




def im2block(image,block): # to divide the image into blocks
		image_block = []
		block_height = block[1]
		block_width = block[0]
		block_size = block_height * block_width
		for j in range(0, image.shape[1], block_height):
				for i in range(0, image.shape[0], block_width):
						image_block.append(image[i:i+block_width, j:j+block_height])
		image_block = np.asarray(image_block).astype(float)
		return image_block

def block2im(mtx, image_size, block): # to combine the blocks back into image
		p, q = block
		sx = image_size[1]
		sy = image_size[0]
		result = np.zeros(image_size)  
		block_index = 0
		for j in range(0,sx,q):
				 for i in range(0,sy,p):
						 result[i:i+q, j:j+p] = mtx[block_index]
						 block_index += 1
		return result


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



def block_compressor(image_blocks,order,method):
	image_blocks_compressed = []

	for image_block in image_blocks:

		if method == 'svd':
			image_block_compress = svd_compression_block(image_block,order)

		elif method == 'fourrier':
			image_block_compress = fourrier_compression_block(image_block,order)
		
		elif method == 'dct':
			image_block_compress = dct_compression_block(image_block,order)
		
		image_blocks_compressed.append(image_block_compress)

	image_compressed = np.stack(image_blocks_compressed)

	return image_compressed


def compressor(image, order, block, method):

	image = image.astype(np.double)

	block_size = block[0] * block[1]
	image_blocks = im2block(image,block)

	image_blocks = block_compressor(image_blocks,order, method)
	
	image_comp = block2im(image_blocks, (image.shape[0],image.shape[1]), block)

	return image_comp


def compress_rgb(img_rgb, order = 5, block = (8,8), method = 'fourrier'):

    img_c_stack = []
    for i in range(3):
            img_c = img_rgb[:,:,i] 
            img_c_compressed = compressor(img_c,order,block = block, method = method)
            img_c_stack.append(img_c_compressed)

    img_rgb = np.dstack(img_c_stack)
    img_rgb = np.around(img_rgb).astype(int)

    return img_rgb