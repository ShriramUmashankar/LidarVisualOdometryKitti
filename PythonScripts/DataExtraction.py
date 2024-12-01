import cv2
import numpy as np
import os 

StereoLeftPath ='/home/shriram/Documents/VisualOdometry/KittiProject/data_odometry_gray/dataset/sequences/00/image_0/'
StereoRightPath = '/home/shriram/Documents/VisualOdometry/KittiProject/data_odometry_gray/dataset/sequences/00/image_1/'

P0 = np.array([7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00, 
                0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00, 
                0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]).reshape(3,4) 
P1 = np.array([7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.861448000000e+02, 
                0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00, 
                0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]).reshape(3,4) 


def CameraParameter():
    k1,r1,t1, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
    t1 = t1/t1[3]

    baseline = t1[0]
    intrinsicMatrix = k1
    focalLength = k1[0][0]

    return baseline,focalLength,intrinsicMatrix

#print(P0)
#print(P1)
# List to store the image file names
image_names = []

# Loop through the files in the folder
for filename in os.listdir(StereoLeftPath):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Add more formats if needed
        image_names.append(filename)

image_names.sort(key=lambda f: int(os.path.splitext(f)[0]))

leftImg = cv2.imread(StereoLeftPath+image_names[0],0)
rightImg = cv2.imread(StereoRightPath+image_names[0],0)
image = cv2.vconcat([leftImg,rightImg])

 # Initialize the SGBM matcher
min_disp = 0
num_disp = 16*5  # Must be divisible by 16
block_size = 5   # Matched block size

# Create SGBM object
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 3 * block_size ** 2,
    P2=32 * 3 * block_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)


disparity = stereo.compute(leftImg, rightImg).astype(np.float32) / 16.0

# Normalize the disparity for visualization (optional)
disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_normalized = cv2.applyColorMap(np.uint8(disparity_normalized),cv2.COLORMAP_JET)

l1 = leftImg
l2 = cv2.imread(StereoLeftPath+image_names[1],0)

consecutive = cv2.vconcat([l1, l2])

orb = cv2.ORB_create()
l1KeyPoints, l1Des = orb.detectAndCompute(l1, None)
l2KeyPoints, l2Des = orb.detectAndCompute(l2, None)


KeyPointsDrawL1 = cv2.drawKeypoints(l1,l1KeyPoints,None)
KeyPointsDrawL2 = cv2.drawKeypoints(l2,l2KeyPoints,None)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(l1Des,l2Des)
matches = sorted(matches, key = lambda x:x.distance)

FeatureMatch = cv2.drawMatches(l1,l1KeyPoints,l2,l2KeyPoints,matches[:50],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)



#cv2.imshow('FeatureMatch', FeatureMatch)
#cv2.imshow('Key points', KeyPointsDrawL1)
#cv2.imshow('Consecutive Move', consecutive)
cv2.imshow('Disparity Map', disparity_normalized)    
#cv2.imshow('Image Pair',image)

cv2.waitKey(0)
cv2.destroyAllWindows()