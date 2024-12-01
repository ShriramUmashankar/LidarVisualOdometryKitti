import numpy as np
import pandas as pd
import cv2
import os
import csv
import matplotlib.pyplot as plt

class DataHandler:
    StereoPath ='/home/shriram/Documents/VisualOdometry/KittiProject/data_odometry_gray/dataset/sequences/'
    GroundTruth = '/home/shriram/Documents/VisualOdometry/KittiProject/data_odometry_poses/dataset/poses/'
    
    image_names = []
    intrinsicMatrix = []
    LeftPath = ''
    RightPath = ''
    baseLine = 0
    FocalLength = 0 
    name = ''

    def __init__(self,name):
        self.name = name
        self.LeftPath = self.StereoPath + name + '/image_0/'
        self.RightPath = self.StereoPath + name + '/image_1/'

        for filename in os.listdir(self.LeftPath):
            if filename.endswith('.jpg') or filename.endswith('.png'): 
                self.image_names.append(filename)

        self.image_names.sort(key=lambda f: int(os.path.splitext(f)[0]))

        calib = pd.read_csv(self.StereoPath + name + '/calib.txt', delimiter = ' ', header = None, index_col = 0)
        P0 = np.array(calib.loc['P0:']).reshape(3,4)
        P1 = np.array(calib.loc['P1:']).reshape(3,4)

        k0,r0,t0, _, _, _, _ = cv2.decomposeProjectionMatrix(P0)
        t0 = t0/t0[3]

        k1,r1,t1, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
        t1 = t1/t1[3]

        self.baseLine = np.round(t1[0][0] - t0[0][0],4)
        self.intrinsicMatrix = k1
        self.FocalLength = k1[0][0]

        return None
    
    def viewStereoPair(self,index):
        leftImg = cv2.imread(self.LeftPath + self.image_names[index],0)
        rightImg = cv2.imread(self.RightPath + self.image_names[index],0)
        image = cv2.vconcat([leftImg,rightImg])
        cv2.imshow("Stereo Pair" , image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def viewDepth(self,index):
        min_disp = 0
        num_disp = 16*5  # Must be divisible by 16
        block_size = 5 

        leftImg = cv2.imread(self.LeftPath + self.image_names[index],0)
        rightImg = cv2.imread(self.RightPath + self.image_names[index],0)

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

        disparity = stereo.compute(leftImg, rightImg).astype(np.float32)/16 
        disparity[disparity == 0] = 0.1
        disparity[disparity == -1] = 0.1

        depthMatrix = self.FocalLength*self.baseLine/disparity

        #depthJetMap = cv2.applyColorMap(np.uint8(depthMatrix),cv2.COLORMAP_JET)
        #disparityJetMap = cv2.applyColorMap(np.uint8(disparity),cv2.COLORMAP_JET)

        #cv2.imshow("Depth Map" , depthJetMap)
        #cv2.imshow("Disparity Map" , disparityJetMap)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return depthMatrix

    def viewForwardMotion(self,index):
        Img1 = cv2.imread(self.LeftPath + self.image_names[index],0)
        Img2 = cv2.imread(self.LeftPath + self.image_names[index + 1],0)
        image = cv2.vconcat([Img1,Img2])

        cv2.imshow("Next Frame to Previous" , image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def groundTruthPose(self,index):
        path = self.GroundTruth + self.name +'.txt'
        load = np.loadtxt(path, delimiter = ' ')
        pose = load[index].reshape(3,4)

        return pose


class MotionEstimation(DataHandler):
    
    T_tot = []

    def __init__(self,name):
        super().__init__(name)
        self.T_tot = np.eye(4)
        print("Class Successfully Initialized")

        return None

    def featureMatch(self,index):
        Img1 = cv2.imread(self.LeftPath + self.image_names[index],0)
        Img2 = cv2.imread(self.LeftPath + self.image_names[index+1],0)
        # cv2.imshow("image" , cv2.vconcat([Img1,Img2]))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(Img1, None)
        kp2, des2 = orb.detectAndCompute(Img2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)

        # FeatureMatch = cv2.drawMatches(Img1,kp1,Img2,kp2,matches[:50],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # cv2.imshow('FeatureMatch', FeatureMatch)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return kp1, kp2, matches    

    def pnp_ransac(self,matches, kp1, kp2, depthMatrix ,max_depth = 30):
        rmat = np.eye(4)
        tvec = np.zeros((3,1))

        img1_points = np.float32([kp1[m.queryIdx].pt for m in matches])
        img2_points = np.float32([kp2[m.trainIdx].pt for m in matches])

        k = self.intrinsicMatrix
        cx = k[0,2]
        cy = k[1,2]
        fx = k[0,0]
        fy = k[1,1]

        object_points = np.zeros((0,3))
        delete = [] #hold entry of feature points with depth more than threshold

        for i, (u,v) in enumerate(img1_points):

            # u,v in np and cv are flipped 
            z = depthMatrix[int(round(v)),int(round(u))]

            if z > max_depth:
                delete.append(i)
                continue

            x = z*(u -cx)/fx
            y = z*(v - cy)/fy  

            object_points = np.vstack([object_points, np.array([x,y,z])])


        img1_points = np.delete(img1_points, delete , 0)
        img2_points = np.delete(img2_points, delete , 0)   

        _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, img2_points, k, None)

        rmat = cv2.Rodrigues(rvec)[0]
        return rmat.round(4),tvec.round(4)  

    def plot(self):
        data = np.loadtxt('position_data.csv', delimiter=',')
        estimated_pos = data[:, 9:12]  
        ground_pos = data[:, 12:]     

        plt.figure(figsize=(20,15))
        plt.plot(estimated_pos[:, 0], estimated_pos[:, 2], lw=3, label='Estimated', color='b')
        plt.plot(ground_pos[:, 0], ground_pos[:, 2], lw=3, label='Ground Truth', color='r')

        # Set axis labels
        plt.xlabel('X Position')
        plt.ylabel('Z Position')

        # Add a legend
        plt.legend()
        plt.grid()

        # Display the plot
        plt.show()

        return None      

    def motion_estimator(self,SaveData,PlotData):
        #for i in range(len(self.image_names)-10):
        for i in range(1000):    
            kp1, kp2, matches = self.featureMatch(i)
            depthMatrix = self.viewDepth(i)
            Pose = self.groundTruthPose(i).round(4)

            estimated_pos = self.T_tot[:3,3]
            estimated_rot = self.T_tot[:3,:3]
            ground_pos = Pose[:,3]

            if SaveData == True :
                with open('position_data.csv', 'a', newline='') as file:
                    writer = csv.writer(file)

                    # In the loop, for each pair of estimated_pos and ground_pos
                    # assuming they are 3x1 numpy arrays
                    writer.writerow(estimated_rot.flatten().tolist()+estimated_pos.flatten().tolist() + ground_pos.flatten().tolist())    

            rmat,tvec = self.pnp_ransac(matches, kp1, kp2, depthMatrix ,max_depth = 30) 

            Tmat = np.eye(4)
            Tmat[:3,:3] = rmat
            Tmat[:3,3] = tvec.T 

            self.T_tot = self.T_tot.dot(np.linalg.inv(Tmat))  

        if PlotData == True:
            self.plot()

        return self.T_tot  






