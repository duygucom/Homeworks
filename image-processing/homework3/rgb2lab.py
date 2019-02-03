from __future__ import division # (if required)

import numpy as np
import cv2

import math

from PIL import Image

from skimage.morphology import erosion, dilation, opening, closing
from skimage import io, color

from matplotlib import pyplot as plt


def p_rgb2lab(orgImg):
	# Create array of image using numpy
	"""
	w = 332
	h = 590

	L_array = []
	a_array = []
	b_array = []

	newlist = []
	"""

	srcArray = np.asarray(orgImg)
	#srcArray = srcArray[...,[0,2,1]]						# manipualte the channels
	# Convert array from RGB into Lab
	srcArray = color.rgb2lab(srcArray)						# get the converted lab values
	"""
	for i in range(w):
		for j in range(h):
			L_array.append(int(srcArray[i][j][0]))
			a_array.append(int(srcArray[i][j][1]))
			b_array.append(int(srcArray[i][j][2]))

	for i in range(w*h):
		newlist.insert(i, (int(L_array[i]), int(a_array[i]), int(b_array[i])))		
																			# convert integer for correct format
	newimage = Image.new(mode="RGB", size = (w,h))							# create a new RGB image

	newimage.putdata(newlist)
	labImg = np.asarray(newimage)				# convert to np array
	"""
	
	final = Image.fromarray(srcArray.astype("uint8"))		# uint8 to get the full range 0-255
	final.save("lab.png")
	
	"""
	plt.hist(a_array,256,[-127,127]) 
	plt.title("a value")
	plt.show()
	plt.hist(b_array,256,[-127,127]) 
	plt.title("b value")
	plt.show()
	plt.hist(L_array,100,[0,100]) 
	plt.title("L value")
	plt.show()
	"""


def opening(img):	

	kernel = np.ones((2,2),np.uint8)						# kernel size = 2x2 for erosion
	kernel2 = np.ones((2,2),np.uint8)						# kernel size = 2x2 for dilation

	erosion = cv2.erode(img,kernel,iterations = 1)			# erode the img with 1 iterarion
	opening = cv2.dilate(erosion,kernel2,iterations = 1)

	cv2.imwrite("erosion.png", erosion)
	cv2.imwrite("opening.png", opening)	



def closing(img):

	kernel = np.ones((2,2),np.uint8)						# kernel size = 2x2 for dilation
	kernel2 = np.ones((2,2),np.uint8)						# kernel size = 2x2 for erosion

	dilosion = cv2.dilate(img,kernel,iterations = 1)		# dilote the img with 1 iterarion
	closing = cv2.erode(dilosion,kernel2,iterations = 1)

	cv2.imwrite("dilosion.png", dilosion)
	cv2.imwrite("closing.png", closing)	


def getRgb(harray,sarray,varray):

	# below algorithm was retrieved from 
	# https://www.rapidtables.com/convert/color/hsv-to-rgb.html

	rval = []
	gval = []
	bval = []

	for i in range(len(harray)):
		# When 0 ≤ H < 360, 0 ≤ S ≤ 1 and 0 ≤ V ≤ 1:

		h1 = (harray[i] / 255) * 360  		# scale it as 0 ≤ H < 360
		s1 = sarray[i] / 255 				# scale it as 0 ≤ S ≤ 1
		v1 = varray[i] / 255 				# scale it as 0 ≤ V ≤ 1

		c = s1 * v1
		x = c * (1 - abs(((h1 / 60) % 2) - 1))
		m = v1 - c

		if 0 <= h1 < 60.0:
			(r1,g1,b1) = (c,x,0)
		elif h1 < 120.0:
			(r1,g1,b1) = (x,c,0)
		elif h1 < 180.0:
			(r1,g1,b1) = (0,c,x)
		elif h1 < 240.0:
			(r1,g1,b1) = (0,x,c)
		elif h1 < 300.0:
			(r1,g1,b1) = (x,0,c)
		elif h1 < 360.0:
			(r1,g1,b1) = (c,0,x)

		(r,g,b) = ((r1 + m) * 255, (g1 + m) * 255, (b1 + m) * 255)
		
		rval.insert(i,r)		# insert r to ith index
		gval.insert(i,g)		# insert g to ith index
		bval.insert(i,b)		# insert b to ith index

	return (rval,gval,bval)

def getHsv(rarray, garray, barray):

	# below algorithm was retrieved from 
	# https://www.rapidtables.com/convert/color/rgb-to-hsv.html


	hval = []
	sval = []
	vval = []

	cmax = 0

	for i in range(len(rarray)):
		r1 = rarray[i] / 255
		g1 = garray[i] / 255
		b1 = barray[i] / 255

		cmax = max(r1,g1,b1)
		cmin = min(r1,g1,b1)

		delta = cmax - cmin

		if delta == 0:
			h1 = 0
		elif cmax == r1:
			h1 = (60 * (((g1-b1)/delta) % 6))
		elif cmax == g1:
			h1 = (60 * (((b1-r1)/delta) + 2))
		elif cmax == b1:
			h1 = (60 * (((r1-g1)/delta) + 4))

		if cmax == 0:
			s1 = 0
		else:
			s1 = delta / cmax

		v1 = cmax

		h1 = (h1 /360) * 255
		s1 = s1 * 255
		v1 = v1 * 255

		hval.insert(i,h1)		# insert hue to ith index
		sval.insert(i,s1)		# insert saturation to ith index
		vval.insert(i,v1)		# insert value to ith index

	return (hval, sval, vval)


def convertRgb(image, size):
	(w, h) = size
	(bval,rval,gval) = image 	# convert rgb to brg

	newlist = []
	
	for i in range(w*h):
		newlist.insert(i, (int(rval[i]), int(gval[i]), int(bval[i])))		# get new data of correct rgb image
																			# convert integer for correct format
	newimage = Image.new(mode="RGB", size = size)							# create a new RGB image

	newimage.putdata(newlist)
	rgbImg = np.asarray(newimage)				# convert to np array

	return rgbImg

def convertHsv(image, size):
	(w, h) = size
	(sval,hval,vval) = image 	# convert hsv to shv

	newlist = []

	for i in range(w*h):
		newlist.insert(i, (int(hval[i]), int(sval[i]), int(vval[i])))		# get new data of correct hsv image
																			# convert integer for correct format

	newimage = Image.new(mode="HSV", size = size)							# create a new HSV image

	newimage.putdata(newlist)
	hsvImg = np.asarray(newimage)				# convert to np array

	return hsvImg

def hsv2rgb(image):

	size = image.size  
	width, height = size  

	(s, h, v) = image.split()					# fixing the channels in order to get the corresponding rgb image. (hsv) -> (shv)
	
	harray = h.getdata() 
	sarray = s.getdata()
	varray = v.getdata()

	rgbVal = getRgb(harray,sarray,varray)		# convert HSV to RGB

	rgbImg = convertRgb(rgbVal, size)			# fixing the channels of RGB image

	restoredImage = Image.fromarray(rgbImg)
	restoredImage.save("boat2.png")


def rgb2hsv(image):					
	
	size = image.size  
	width, height = size 

	(g,b,r) = image.split()						# fixing the channels in order to get the corresponding hsv image. (rgb) -> (gbr)
	rarray = r.getdata() 
	garray = g.getdata()
	barray = b.getdata()

	hsvVal = getHsv(rarray, garray, barray)		# convert RGB to HSV

	hsvImg = convertHsv(hsvVal, size)			# fixing the channels of HSV image

	restoredImage = Image.fromarray(hsvImg)
	restoredImage.save("distorted2.png")
"""
def gaussian(image):

	row,col,ch= image.shape
	mean = 0
	gauss = np.random.normal(mean,20,(row,col,ch))
	gauss = gauss.reshape(row,col,ch)
	noisy = image + gauss
	cv2.imwrite("gaussian.png", noisy)
	return noisy

def saltnpepper(image):
	row,col,ch = image.shape
	s_vs_p = 0.5
	amount = 0.04
	out = image
	# Salt mode
	num_salt = np.ceil(amount * image.size * s_vs_p)
	coords = [np.random.randint(0, i - 1, int(num_salt))
			  for i in image.shape]
	out[coords] = 1

	# Pepper mode
	num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
	coords = [np.random.randint(0, i - 1, int(num_pepper))
			  for i in image.shape]
	out[coords] = 0
	cv2.imwrite("saltnpepper.png", out)
	return out
"""	
if __name__ == '__main__':

	distortedImg = Image.open("distorted.png")  # read distorted image
	orgImg = Image.open("boat.png") 			# read original image
	
	# converting hsv and rgb images to each other
	hsv2rgb(distortedImg)						# convert distorted image to the original
	rgb2hsv(orgImg)								# convert original image to the distorted
	
	# morphological filter
	rgb2 = cv2.imread("boat2.png")				# read retrieved rgb image
	opening(rgb2)								# erosion + dilarion were applied on the retrieved image
												# in order to fix the ruptures on it
	
	hsv2 = cv2.imread("distorted2.png")			# read retrieved hsv image
	closing(hsv2)

	# original rgb image to lab
	p_rgb2lab(orgImg)

	"""
	# plot the histogram
	img = cv2.imread("lab1.png")

	plt.hist(img.ravel(),256,[-127,127]) 
	plt.show()
	"""
	
	"""
	#histogram equalization
	
	gray = cv2.imread("boat.png",0)
	cv2.imwrite("gray.png",gray)
	equ = cv2.equalizeHist(gray)
	cv2.imwrite("equ.png",equ)
	res = np.hstack((gray,equ)) #stacking images side-by-side
	cv2.imwrite('res.png',res)
	"""
	"""
	# fix the retrieved image with median filter
	median = cv2.medianBlur(rgb2,7)				# median blurring of the retrieved rgb image with 7x7 kernel size
	cv2.imwrite("medianBlur7x7.png", median)

	# previous image operations
	# gray = cv2.imread("boat2.png",0)			# convert the retrieved image to grayscale
	original = cv2.imread("boat.png")
	
	# OpenCV library was used for averaging and median filtering:
	# https://bit.ly/2s6K579

	blur = cv2.blur(original,(3,3))				# averaging on the original image with 3x3 kernel size
	cv2.imwrite("averaging3x3.png", blur)

	#Salt-and-Pepper and Gaussian Noise were implemented based on the code from:
	#https://stackoverflow.com/a/30624520 
	
	gaussNoise = gaussian(original)				# gaussian noise added to the original image
	snpNoise = saltnpepper(original)			# salt and pepper noise added to the original imafe

	medianSalt = cv2.medianBlur(snpNoise,3)				# median blurring of the salt and pepper image with 3x3 kernel size
	cv2.imwrite("medianSalt.png", medianSalt)
	
	medianGauss = cv2.medianBlur(gaussNoise.astype(np.float32),3)				# median blurring of the gaussian image with 3x3 kernel size
	cv2.imwrite("medianGauss.png", medianGauss)	
	"""