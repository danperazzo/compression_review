import numpy as np
from scipy import fftpack

from abc import ABC, abstractmethod

def compression_factory(method_type, order, block_size):

	if method_type == 'dft':
		return DFTCompression(order, block_size)
	elif method_type == 'dct':
		return DCTCompression(order, block_size)
	elif method_type == 'pca':
		return PCACompression(order, block_size)
	else:
		return SVDCompression(order, block_size)



class BaseBlockCompression(ABC):
	def __init__(self, order, block_size):
		self.order = order
		self.block_size = block_size
	
	@abstractmethod
	def compression_block(self,image):
		pass
	
	# Code based on https://github.com/SonuDileep/KL-transform-for-Image-Data-Compression/blob/master/KL_Transform%20for%20data:image%20compression.ipynb
	def im2block(self, image): # to divide the image into blocks
		image_block = []

		for j in range(0, image.shape[1], self.block_size):
				for i in range(0, image.shape[0], self.block_size):
						image_block.append(image[i:i+self.block_size, j:j+self.block_size])
		image_block = np.asarray(image_block).astype(float)

		return image_block
	
	def block2im(self, mtx, image_size): # to combine the blocks back into image
		sx = image_size[1]
		sy = image_size[0]
		result = np.zeros(image_size)  
		col = 0
		for j in range(0,sx,self.block_size):
				 for i in range(0,sy,self.block_size):
						result[i:i+self.block_size, j:j+self.block_size] = mtx[col]
						col += 1
		return result
	
	def block_compressor(self,image_block_list):
		image_blocks_compressed = []
		for image_block in image_block_list:

			image_compressed = self.compression_block(image_block)
			image_blocks_compressed.append(image_compressed)

		image_compressed = np.stack(image_blocks_compressed)

		return image_compressed
	
	def compreses_channel(self,image):
		image = image.astype(np.double)
		image_blocks = self.im2block(image)
		image_compressed = self.block_compressor(image_blocks)
		

		image_comp = self.block2im(image_compressed, (image.shape[0],image.shape[1]))

		return image_comp
	

	def compress_rgb(self, img_rgb):

		img_c_stack = []
		for i in range(3):
				img_c = img_rgb[:,:,i] 
				img_c_compressed = self.compreses_channel(img_c)
				img_c_stack.append(img_c_compressed)

		img_rgb = np.dstack(img_c_stack)
		img_rgb = np.around(img_rgb).astype(int)

		return img_rgb
		
	
class DFTCompression(BaseBlockCompression):

	def compression_block(self, image):
		image_dft = np.fft.fft2(image)

		image_filter = np.zeros(image.shape)

		num_freq = int(self.block_size*((self.order)**0.5))
		image_filter[:num_freq,:num_freq] = image_dft[:num_freq,:num_freq]
		
		image_compressed = np.fft.ifft2(image_filter).real

		return image_compressed
	
class DCTCompression(BaseBlockCompression):

	# Code for transforms from: http://www.jeanfeydy.com/Teaching/MasterClass_Radiologie/Part%206%20-%20JPEG%20compression.html
	def dct2(self,f):
		"""
		Discrete Cosine Transform in 2D.
		"""
		return np.transpose(fftpack.dct(
			np.transpose(fftpack.dct(f, norm = "ortho")), norm = "ortho"))

	def idct2(self,f):
		"""
		Inverse Discrete Cosine Transform in 2D.
		"""
		return np.transpose(fftpack.idct(
			np.transpose(fftpack.idct(f, norm = "ortho")), norm = "ortho"))

	def compression_block(self, image):	
		image_dct = self.dct2(image)

		image_filter = np.zeros(image_dct.shape)
		num_freq = int(self.block_size*((self.order)**0.5))
		image_filter[:num_freq,:num_freq] = image_dct[:num_freq,:num_freq]
		
		image_compressed = self.idct2(image_filter)

		return image_compressed
	

class PCACompression(BaseBlockCompression):

	def compression_block(self, image):
		size = image.shape[1]

		mean = np.mean(image,axis=0) 
		image = image - mean

		# Decorrelate columns of image
		covariance = np.cov(image, rowvar=False)
		_, eig = np.linalg.eigh(covariance) # Eigenvectors ordered low-to-high 
		PCA = image @ eig

		# Quantization of the transformed image
		D = int(self.order*self.block_size)
		PCA_compressed = PCA[:,size-D:size]
		eig_compressed = eig[:,size-D:size] 

		image_comp = (PCA_compressed @ np.transpose(eig_compressed)) + mean 
		return image_comp


class SVDCompression(BaseBlockCompression):

	def compression_block(self, image):
		D = int(self.order*self.block_size)
		[Umat,Smat,Vmat] = np.linalg.svd(image, full_matrices=False, compute_uv=True, hermitian=False)

		S_compressed = Smat[:D]
		U_compressed = Umat[:,:D]
		V_compressed = Vmat[:D,:]

		img_compressed = np.dot(U_compressed * S_compressed, V_compressed)
		return np.round(img_compressed)	







