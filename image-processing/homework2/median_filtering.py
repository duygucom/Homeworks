from cv2 import * #Import functions from OpenCV
import numpy as np
import cv2

"""

def float16_to_std_uint8(img):
	# Properly handles the conversion to uint8
	img = cv2.convertScaleAbs(img, alpha = (255/1))
	 
	return img

def to_std_float(img):
	#Converts img to 0 to 1 float to avoid wrapping that occurs with uint8
	img.astype(np.float16, copy = False)
	img = np.multiply(img, (1/255))
	 
	return img


def salt_n_pepper(img, pad = 15, show = 1):
	# https://blog.kyleingraham.com/2017/02/04/salt-pepper-noise-and-median-filters-part-ii-the-code/
	# Convert img1 to 0 to 1 float to avoid wrapping that occurs with uint8
	img = to_std_float(img)
	 
	# Generate noise to be added to the image. We are interested in occurrences of high
	# and low bounds of pad. Increased pad size lowers occurence of high and low bounds.
	# These high and low bounds are converted to salt and pepper noise later in the
	# function. randint is inclusive of low bound and exclusive of high bound.
	noise = np.random.randint(pad, size = (img.shape[0], img.shape[1]))
	 
	# Convert high and low bounds of pad in noise to salt and pepper noise then add it to
	# our image. 1 is subtracted from pad to match bounds behaviour of np.random.randint.
	img = np.where(noise == 0, 0, img)
	img = np.where(noise == (pad-1), 1, img)
	 
	# Properly handles the conversion from float16 back to uint8
	img = float16_to_std_uint8(img)

	print(img.dtype)

	#cv2.imshow('Image with Salt & Pepper Noise', img)
	cv2.imwrite('salt_n_pepper.png',img)

	return img

def float64_to_std_uint8(img):
	img = img.astype(np.uint8)
	return img

def gaussian(image):
	#http://www.magikcode.com/?p=240
	temp_image = np.float64(np.copy(image))

	row,col= image.shape
	sigma = 20

	noise = np.random.randn(row,col)*sigma
	noisy_image = np.zeros(temp_image.shape,np.float64)
	noisy_image = temp_image + noise

	noisy_image = float64_to_std_uint8(noisy_image)


	cv2.imshow('gauss',noisy_image)
	cv2.waitKey(0)
	return noisy_image
"""


if __name__ == '__main__':

	source = cv2.imread("salt_n_pepper.png", 0)
	
	"""
	gaussian(source)
	salt_n_pepper(source)
	"""
	#cv2.imshow('Source_Picture', source)

	final = source

	tmp = [None]*9	#in order to sort the 9 neighborhoud pixels.
	
	for y in range(0,source.shape[0]-1):
		for x in range(0,source.shape[1]-1):
			# assign 0 to the values on the borders
			if y == source.shape[0]-2:
				final[y+1,x] = 0
			if x == source.shape[1]-2:
				final[y,x+1] = 0
			# get 9 pixels and sort
			tmp[0] = source[y-1,x-1]
			tmp[1] = source[y,x-1]
			tmp[2] = source[y+1,x-1]
			tmp[3] = source[y-1,x]
			tmp[4] = source[y,x]
			tmp[5] = source[y+1,x]
			tmp[6] = source[y-1,x+1]
			tmp[7] = source[y,x+1]
			tmp[8] = source[y+1,x+1]
			
			tmp.sort()
			#assign the median value to the center.
			final[y,x]=tmp[4]
			
	cv2.imshow('Final_Picture', final) #Show the image
	cv2.waitKey()
	
	""" 
	tmp=[None]*25
	
	for y in range(0,source.shape[0]-2):
		for x in range(0,source.shape[1]-2):
			tmp[0] = source[y-2,x-2]
			tmp[1] = source[y-1,x-2]
			tmp[2] = source[y,x-2]
			tmp[3] = source[y+1,x-2]
			tmp[4] = source[y+2,x-2]
			tmp[5] = source[y-2,x-1]
			tmp[6] = source[y-1,x-1]
			tmp[7] = source[y,x-1]
			tmp[8] = source[y+1,x-1]
			tmp[9] = source[y+2,x-1]
			tmp[10] = source[y-2,x]
			tmp[11] = source[y-1,x]
			tmp[12] = source[y,x]
			tmp[13] = source[y+1,x]
			tmp[14] = source[y+2,x]
			tmp[15] = source[y-2,x+1]
			tmp[16] = source[y-1,x+1]
			tmp[17] = source[y,x+1]
			tmp[18] = source[y+1,x+1]
			tmp[19] = source[y+2,x+1]
			tmp[20] = source[y-2,x+2]
			tmp[21] = source[y-1,x+2]
			tmp[22] = source[y,x+2]
			tmp[23] = source[y+1,x+2]
			tmp[24] = source[y+2,x+2]

			
			tmp.sort()
			final[y,x]=tmp[12]
	cv2.imshow('Final_Picture', final) #Show the image
	cv2.waitKey()
	"""       
