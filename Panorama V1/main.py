import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
import matplotlib.image as im
import itertools
import sys

def findKeyPoints(img, template, distance=200):


    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img,None)
    kp2, des2 = sift.detectAndCompute(template,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    skp_final = []
    tkp_final = []
    for m,n in matches:
        skp_final.append(m)
        tkp_final.append(n)
        if m.distance < 0.75*n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img,kp1,template,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()

    return 
def remon(self, image):
		# convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
		# check to see if we are using OpenCV 3.X
        if self.isv3:
			# detect and extract features from the image
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)
 
		# otherwise, we are using OpenCV 2.4.X
        else:
			# detect keypoints in the image
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)
 
			# extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)
 
		# convert the keypoints from KeyPoint objects to NumPy
		# arrays
        kps = np.float32([kp.pt for kp in kps])
 
		# return a tuple of keypoints and features
        return (kps, features)

def drawKeyPoints(img, template, skp, tkp, num=-1):
    pts = []
    h1, w1 = img.shape[:2]
    h2, w2 = template.shape[:2]
    nWidth = w1+w2
    nHeight = max(h1, h2)
    hdif = (h1-h2)/2
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
    newimg[hdif:hdif+h2, :w2] = template
    newimg[:h1, w2:w1+w2] = img

    maxlen = min(len(skp), len(tkp))
    if num < 0 or num > maxlen:
        num = maxlen
    for i in range(num):
        pt_a = (int(tkp[i].pt[0]), int(tkp[i].pt[1]+hdif))
        pt_b = (int(skp[i].pt[0]+w2), int(skp[i].pt[1]))
        pts.append(pt_a)
        pts.append(pt_b)
        print pt_a
        print pt_b
    return pts


def match(imageA,imageB):
    img = imageA
    temp = imageB

    dist = 200
    num = 4
    # (kpsA, featuresA) = self.detectAndDescribe(img)
    # (kpsB, featuresB) = self.detectAndDescribe(temp)
    skp, tkp = findKeyPoints(img, temp, dist)
    return drawKeyPoints(img, temp, skp, tkp, num)

if __name__=="__main__":
    
    fileA = "imgs/pic2.jpg"
    fileB = "imgs/pic1.jpg"

    imageA = im.imread(fileA)
    imageB = im.imread(fileB)

    # points needed to be selected with mouse
    numPoints = 16
    # an array holding x,y coordinates of manually entered points
    pts= []
    mode = raw_input("for sift enter sift: ")
    pts = []
    if (mode == "sift"):
        pts = match(imageA,imageB)

    else:
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
        # print("Before")
        # print(pts)



    pts = np.reshape(pts, (int(numPoints/2),4))
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

    H=[[a,b,c],[d,e,f],[g,h,1]]

    #verifining the H matrix
    fig = plt.figure()
    figB = fig.add_subplot(1,2,2)
    figA = fig.add_subplot(1,2,1)
    figB.imshow(imageB,origin='upper')
    figA.imshow(imageA,origin='upper')
    plt.axis('image')
    i = 0
    while (i < (int(numPoints/2))):
        pts = plt.ginput(1,timeout=0)
        pts = np.reshape(pts,(1*2,1))
        toTrans = np.ones((3,1))
        toTrans[0][0] = pts[0]
        toTrans[1][0] = pts[1]
        p = np.dot(H,toTrans)
        x = p[0][0]/p[2][0]
        y = p[1][0]/p[2][0]
        figB.scatter([x],[y])
        i = i + 1




    ## warping between image plane

    img2 = cv2.imread(fileA)
    img1 = cv2.imread(fileB)
    mv1 = []
    mv2 = []
    rows = img1.shape[0]
    cols = (img2.shape[1] + (img2.shape[1]))
    results = [np.zeros((rows,cols),np.uint8),np.zeros((rows,cols),np.uint8),np.zeros((rows,cols),np.uint8)]
    mv1 = cv2.split(img1,mv1)
    mv2 = cv2.split(img2,mv2)
    H = np.array(H,np.float64)
    #loop on RGB
    for ii in range(0,3):
        img1 =mv1[ii]
        img2 = mv2[ii]



        #H = [[5.93314812e-01,6.41782199e-02,2.64710654e+02], [ -2.70368012e-01,7.92183580e-01,5.89275035e+01], [ -7.01814541e-04,-2.20932155e-04,1.00000000e+00]]
        #H = [[0.585512,0.128589,259.396],[-0.285436,0.86002,52.1516],[-0.000748086,-0.000101119,1]]
        #H2 = [[1.5995,-0.28828,-399.5444],[0.455588717,1.438633,-192.9034],[0.00124,math.pow(10,-5)*-7.51721074934485,1]]
        Hinv = np.linalg.inv(H)
        pixel = np.ones((3,1))
        transPix = np.zeros((3,1),np.float64)
    

        #print pixel
        for i in range(0,img1.shape[0]):    #loop on y
            for j in range(0,img1.shape[1]):    #loop on x
                pixel[0][0] = j
                pixel[1][0] = i
                pixel[2][0] = 1

                transPix = np.dot(Hinv,pixel)
                x = transPix[0][0] / transPix[2][0]
                y = transPix[1][0] / transPix[2][0]
                l = int(math.floor(x))
                k = int(math.floor(y))


                if(k< results[ii].shape[0]and l < results[ii].shape[1] and k >=0 and l>=0):
                    results[ii][k][l] = img1[i][j]
                    #fill holes using inverse wrapping
                    invWrap = np.zeros((3,1),np.float64)
                    uprow = np.int(k-1)
                    leftcol = np.int(l-1)
                    downrow = np.int(k+1)
                    rightcol = np.int(l+1)
                    for r in range(uprow,downrow):
                        for c in range(leftcol,rightcol):
                            if (r == k and c == l):
                                continue
                            if(r>0 and r <results[ii].shape[0] and c > 0 and c < results[ii].shape[1]):
                                invWrap[0][0] = c
                                invWrap[1][0] = r
                                invWrap[2][0] = 1
                                invWrap = np.dot(Hinv,invWrap)
                                x = invWrap[0][0] / invWrap[2][0]
                                x= int(x)
                                y = invWrap[1][0] / invWrap[2][0]
                                y = int(y)
                                if((x < img1.shape[1] and x > 0 and y < img1.shape[0]) and y >0 ):
                                    results[ii][r][c] = img1[y][x]


        for i in range(0,img2.shape[0]):
            for j in range(0,img2.shape[1]):
                results[ii][i][j] = img2[i][j]
        print ("channel done")

        for i in range(0,results[ii].shape[0]):
                for j in range(0,results[ii].shape[1]):
                    if(results[ii][i][j]==0):
                        jj = j
                        while(jj<results[ii].shape[1] and results[ii][i][jj]==0):
                            results[ii][i][jj] = results[ii][i][jj-1]
                            jj = jj +1
                        j=jj
  


    res = cv2.merge(results)
    cv2.imshow("window",res)
    cv2.waitKey(0)