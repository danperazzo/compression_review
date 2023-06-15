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
	def encode_block(self,image):
		pass

	@abstractmethod
	def decode_block(self,image):
		pass
	
	# Code based on https://github.com/SonuDileep/KL-transform-for-Image-Data-Compression/blob/master/KL_Transform%20for%20data:image%20compression.ipynb
	def im2encoded_blocks(self, image): # to divide the image into blocks
		image_block_list = []
		for j in range(0, image.shape[1], self.block_size):
				for i in range(0, image.shape[0], self.block_size):
						image_block = image[i:i+self.block_size, j:j+self.block_size]
						encoded_blocks = self.encode_block(image_block)
						image_block_list.append(encoded_blocks)

		image_block = image_block_list

		return image_block
	
	def encoded_block2im(self, block_list, image_size): # to combine the blocks back into image
		width = image_size[1]
		height = image_size[0]
		result = np.zeros(image_size)  
		index_block = 0
		for j in range(0,width,self.block_size):
				 for i in range(0,height,self.block_size):
						result[i:i+self.block_size, j:j+self.block_size] = self.decode_block(block_list[index_block])
						index_block += 1
		return result
	
	
	def encode_channel(self,image):
		image = image.astype(np.double)
		block_codes = self.im2encoded_blocks(image)

		encoded_image = {'block_codes':block_codes, 'orig_dims':image.shape}
		return encoded_image
	
	def decode_channel(self,encoded_image):

		block_codes = encoded_image['block_codes']
		orig_dims = encoded_image['orig_dims']

		image_comp = self.encoded_block2im(block_codes, orig_dims)

		return image_comp
	

	def encode_rgb(self, img_rgb):

		img_encoded_rgb_list = []
		for i in range(3):
				img_c = img_rgb[:,:,i] 
				img_c_coded = self.encode_channel(img_c)
				img_encoded_rgb_list.append(img_c_coded)

		return img_encoded_rgb_list
	
	def decode_rgb(self, img_encoded_rgb_list):
		
		img_decoded_rgb_list = []
		for i in range(3):
			img_c_decoded = self.decode_channel(img_encoded_rgb_list[i])
			img_decoded_rgb_list.append(img_c_decoded)

		img_rgb = np.dstack(img_decoded_rgb_list)
		img_rgb = np.around(img_rgb).astype(int)

		return img_rgb
		
	
class DFTCompression(BaseBlockCompression):

	def encode_block(self, image):
		image_dft = np.fft.fft2(image)

		num_freq = int(self.block_size*((self.order)**0.5))
		image_compressed = image_dft[:num_freq,:num_freq]

		code_dict = {'image_compressed':image_compressed, 'orig_dim':image.shape}

		return code_dict
	
	def decode_block(self, code_dict):
		image_compressed = code_dict['image_compressed']
		orig_dim = code_dict['orig_dim']

		num_freq = image_compressed.shape[0]

		image_reconstruction_freq = np.zeros(orig_dim)
		image_reconstruction_freq[:num_freq,:num_freq] = image_compressed
		
		image_compressed = np.fft.ifft2(image_reconstruction_freq).real

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

	def encode_block(self, image):
		image_dct = self.dct2(image)

		num_freq = int(self.block_size*((self.order)**0.5))
		image_compressed = image_dct[:num_freq,:num_freq]

		code_dict = {'image_compressed':image_compressed, 'orig_dim':image.shape}

		return code_dict
	
	def decode_block(self, code_dict):
		image_compressed = code_dict['image_compressed']
		orig_dim = code_dict['orig_dim']

		num_freq = image_compressed.shape[0]

		image_reconstruction_freq = np.zeros(orig_dim)
		image_reconstruction_freq[:num_freq,:num_freq] = image_compressed
		
		image_compressed = self.idct2(image_reconstruction_freq)

		return image_compressed
	

class PCACompression(BaseBlockCompression):
	
	def encode_block(self, image):
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

		code_dict = {'PCA_compressed':PCA_compressed, 'eig_compressed':eig_compressed, 'mean':mean }

		return code_dict
	
	def decode_block(self, code_dict):
		PCA_compressed = code_dict['PCA_compressed']
		eig_compressed = code_dict['eig_compressed']
		mean = code_dict['mean']

		image_comp = (PCA_compressed @ np.transpose(eig_compressed)) + mean 
		return image_comp

class SVDCompression(BaseBlockCompression):
	
	def encode_block(self, image):
		D = int(self.order*self.block_size)
		[Umat,Smat,Vmat] = np.linalg.svd(image, full_matrices=False, compute_uv=True, hermitian=False)


		S_compressed = Smat[:D]
		U_compressed = Umat[:,:D]
		V_compressed = Vmat[:D,:]

		code_dict = {'S_compressed':S_compressed, 'U_compressed':U_compressed, 'V_compressed':V_compressed }

		return code_dict
	
	def decode_block(self, code_dict):
		S_compressed = code_dict['S_compressed']
		U_compressed = code_dict['U_compressed']
		V_compressed = code_dict['V_compressed']


		img_compressed = np.dot(U_compressed * S_compressed, V_compressed)

		return np.round(img_compressed)	







