import matplotlib.pyplot as plt
import numpy as np;
import matplotlib.image as mpimg;
# import get_correspondence as getCorresp;
#import scipy as sc;

from PIL import Image;

def getStartX(pts, index):
	return pts[0][index][0];

def getStartY(pts, index):
	return pts[0][index][1];

def getEndX(pts, index):
	return pts[1][index][0];

def getEndY(pts, index):
	return pts[1][index][1];

def get_sub_A_matrix(x, y, x_t, y_t):
	subMatrix = np.zeros((2,8));
	subMatrix[0] = np.array([x, y,1,0,0,0, -x*x_t, -y*x_t]);
	subMatrix[1] = np.array([0,0,0,x, y,1, -x*y_t, -y*y_t]);
	return subMatrix;

# Transform point using transform h
def transform(point,h):
	h_2d = np.zeros((3,3));
	for i in range(0,3):
		for j in range(0,3):
			if 3*i+j < 8:
				h_2d[i][j] = h[3*i+j];
	h_2d[2][2] = 1;
#	print('h_2d ' + str(h_2d));
	ret = np.dot(h_2d, point);
	ret[0] = ret[0]/ret[2];
	ret[1] = ret[1]/ret[2];
	ret[2] = 1;
	return ret;

def InvTransform(point,h):
	h_2d = np.zeros((3,3));
	for i in range(0,3):
		for j in range(0,3):
			if 3*i+j < 8:
				h_2d[i][j] = h[3*i+j];
	h_2d[2][2] = 1;
#	print('h_2d ' + str(h_2d));
	ret = np.dot(np.linalg.inv(h_2d), point);
	ret[0] = ret[0]/ret[2];
	ret[1] = ret[1]/ret[2];
	ret[2] = 1;
	return ret;


# Manually identifies corresponding points from two views
def getCorrespondence(imageA, imageB,numPoints):
	pts= []

    #display the 2 images to user
	fig = plt.figure()
	figA = fig.add_subplot(1,2,1)
	figB = fig.add_subplot(1,2,2)
    # Display the image
	figB.imshow(imageB,origin='upper')
	figA.imshow(imageA,origin='upper')
	plt.axis('image')
    # plt.show()

    #Manually enter correspondence points
    #user should enter in order the correspondence point in each image
	pts = plt.ginput(numPoints,timeout= 0)
	pts = np.reshape(pts, (int(numPoints/2),4))

	return pts

# calcuate H matrix using n matched points from get_correspondence
def calcH(pts,numPoints):
	xy = pts[:,[2,3]]
	A=np.zeros((numPoints,8),'float64')
	for i in range(int(numPoints/2)):
		A[2*i][0]=pts[i][0]
		A[2*i][1]=pts[i][1]
		A[2*i][2]=1
		A[2*i][6]=-pts[i][0]*pts[i][2]
		A[2*i][7]=-pts[i][1]*pts[i][2]
		A[2*i+1][3]=pts[i][0]
		A[2*i+1][4]=pts[i][1]
		A[2*i+1][5]=1
		A[2*i+1][6]=-pts[i][0]*pts[i][3]
		A[2*i+1][7]=-pts[i][1]*pts[i][3]

	Y=np.reshape(xy,(numPoints,1))
	a,b,c,d,e,f,g,h = np.linalg.lstsq(A, Y)[0]
	hh = np.linalg.lstsq(A, Y)[0]
	H=[[a,b,c],[d,e,f],[g,h,1]]
	return hh;

#========================== NEW WRAP IMAGE

def wrapAndMergeImage(sourceImage , h, refImage):
	# here we return new sourceImage with (2*width,2*height), and transform our image into destination .. with interpolating
	height = sourceImage.shape[0];
	width = sourceImage.shape[1];

	min_mapped_i = int(100000);
	min_mapped_j = int(100000);
	max_mapped_i = max_mapped_j = int(-100000);

	# calculate bounds of new image
	print('Image A size ' + str(sourceImage.shape));


	# calcuate bounds only!
	bounds = np.array([[0,0],[height-1, 0],[0, width-1],[height-1, width-1]]);
	for k in range(0,4):
		i = bounds[k][0]
		j = bounds[k][1];
		# perform multiplication
		mappedPos = transform(np.array([[j],[i],[1]]), h);
		# let's
		mapped_j = int(mappedPos[0][0])
		mapped_i = int(mappedPos[1][0])
		# update bounding box!
		if mapped_i < min_mapped_i:
			min_mapped_i = mapped_i
		if mapped_i > max_mapped_i:
			max_mapped_i = mapped_i
		if mapped_j < min_mapped_j:
			min_mapped_j = mapped_j
		if mapped_j > max_mapped_j:
			max_mapped_j =mapped_j
	
	newHeight = (max_mapped_i-min_mapped_i+1);
	newWidth = (max_mapped_j-min_mapped_j+1);

	shiftHeight = - min_mapped_i;
	shiftWidth = - min_mapped_j;

	print('Bounds ' + str(min_mapped_i) + ' ' + str(max_mapped_i) + ' ' + str(min_mapped_j) + ' ' + str(max_mapped_j));
	print('shiftHeight ' + str(shiftHeight));
	print('shiftWidth ' + str(shiftWidth));


	destinationImage = np.zeros((newHeight,newWidth,3), dtype=np.uint8);

	print('Destination image size ' + str(destinationImage.shape))

	# write to the new image!

#	for i in range(0,height):
#		for j in range(0, width):
#			mappedPos = transform(np.array([[j],[i],[1]]), h);
#			mapped_j = int(mappedPos[0][0])
#			mapped_i = int(mappedPos[1][0])
#			destinationImage[mapped_i+shiftHeight][mapped_j+shiftWidth] = sourceImage[i][j];

#	im = Image.fromarray(destinationImage)
#	im.save("with_holes.jpg")

	# let's Remove black holes
	for i in range(0, newHeight):
		for j in range(0, newWidth):
			# may be done in more neat way!
			if int(destinationImage[i][j][0]) == 0 and int(destinationImage[i][j][1]) == 0 and int(destinationImage[i][j][2]) == 0:
				# it's black let's get back to it's inverse!
				inv_mapped_pos = InvTransform(np.array([[(j - shiftWidth)], [(i - shiftHeight)],[1]]), h)
				inv_mapped_j = inv_mapped_pos[0][0]
				inv_mapped_i = inv_mapped_pos[1][0]
				if inv_mapped_i <= height-1 and  inv_mapped_i >= 0 and inv_mapped_j <= width-1 and inv_mapped_j >= 0:
					# 	using bilinear interpolation!
					low_i = int(inv_mapped_i);
					low_j = int(inv_mapped_j);
					dist_i = inv_mapped_i - low_i;
					dist_j = inv_mapped_j - low_j;
					destinationImage[i][j] = (1-dist_i)*(1-dist_j)*sourceImage[low_i][low_j] + (1-dist_i)*(dist_j)*sourceImage[low_i][low_j+1] + (dist_i)*(1-dist_j)*sourceImage[low_i+1][low_j] + (dist_i)*(dist_j)*sourceImage[low_i+1][low_j+1];
#					destinationImage[i][j] = sourceImage[int(inv_mapped_i)][int(inv_mapped_j)]

	im = Image.fromarray(destinationImage)
	im.save("without_holes.jpg")

	# merge original image!
	ref_image_height = refImage.shape[0]
	ref_image_width = refImage.shape[1]

	print('Ref-image size ' + str(refImage.shape))

	mergedImage_height = ref_image_height + shiftHeight
	if newHeight > mergedImage_height:
		mergedImage_height = newHeight

	mergedImage_width = ref_image_width + shiftWidth
	if newWidth > mergedImage_width:
		mergedImage_width = newWidth

	mergedImage = np.zeros((mergedImage_height, mergedImage_width, 3), dtype=np.uint8);

	# sketch the reference image
	for i in range(0,ref_image_height):
		for j in range(0, ref_image_width):
#			if not( i+ shiftHeight < newHeight and j + shiftWidth < newWidth):
			mergedImage[i + shiftHeight][j + shiftWidth] = refImage[i][j]

	# sketch the destination image
	for i in range(0,newHeight):
		for j in range(0, newWidth):
			if not( int(destinationImage[i][j][0]) == 0 and int(destinationImage[i][j][0]) == 0 and int(destinationImage[i][j][2]) == 0 ):
				mergedImage[i][j] = destinationImage[i][j]

	im = Image.fromarray(mergedImage)
	im.save("after_add_ref_image.jpg")

	return destinationImage;



def verifyH(imageA, imageB, h):
	num = 400;
	imageA_height = imageA.shape[0];
	imageA_width = imageA.shape[1];
	imageB_height = imageB.shape[0];
	imageB_width = imageB.shape[1];

	fig = plt.figure()
	figA = fig.add_subplot(1,2,1)
	figB = fig.add_subplot(1,2,2)
	# Display the image
	# lower use to flip the image
	figA.imshow(imageA)#,origin='lower')
	
	prev = np.array([100000,1000000])

	for k in range(0, num):
		figB.imshow(imageB)#,origin='lower')
		plt.axis('image')
		# n = number of points to read
		pts = plt.ginput(n=1, timeout=0)
		j = pts[0][0]
		i = pts[0][1];
		print(str(i) + ' ' + str(j));
		print(str(i-prev[0]) + ' ,, ' + str(j-prev[1]))
		if abs(i-prev[0]) < 2 and abs(j-prev[1]) < 2:
			return;
		prev[0] = i;
		prev[1] = j;
		if  not(int(i) > -1 and int(i) < imageA_height and int(j) > -1 and int(j) < imageA_width):
			return;
		mappedPos = transform(np.array([[j],[i],[1]]), h);
		mapped_i = mappedPos[1][0];
		mapped_j = mappedPos[0][0];
		for newi in range(int(mapped_i)-2, int(mapped_i)+3):
			for newj in range(int(mapped_j)-2, int(mapped_j)+3):
				if newi > -1 and newi < imageB_height and newj > -1 and newj < imageB_width:
					imageB[int(newi)][int(newj)] = np.array([1,1,1]);

def main():
	fileA = 'imageA.jpg';
	fileB = 'imageB.jpg';
	
	imageA = mpimg.imread(fileA);
	imageB = mpimg.imread(fileB);

	numPoints = 8

	#get correspondence point from user
	pts = getCorrespondence(imageA, imageB,numPoints);

	h = calcH(pts,numPoints);
	print('H :' + str(h));
	print(type(h))
#	print(imageA);
	#wrapping

	wrapAndMergeImage(imageA, h, imageB);
#	verifyH(imageA, imageB, h);

if __name__=="__main__":
	main();