import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from compression_transforms import block_compressor


order = 10

img_rgb = np.array(Image.open('datasets/kodak/kodim01.png'))

plt.imshow(img_rgb.astype(int))
plt.show()
 
#convert the image into grayscale

img_c_stack = []
for i in range(3):
		img_c = img_rgb[:,:,i] 
		img_c_compressed = block_compressor(img_c,order,block = (8,8), compression = 'fourrier')
		img_c_stack.append(img_c_compressed)


img_rgb = np.dstack(img_c_stack)
img_rgb = np.around(img_rgb).astype(int)

plt.imshow(img_rgb)
plt.show()

print(img_rgb.shape)

