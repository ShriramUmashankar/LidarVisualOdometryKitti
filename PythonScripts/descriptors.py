import cv2
import numpy as np
from VisualOdometry import MotionEstimation
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import csv

dataSet = MotionEstimation('00')
image_names = dataSet.image_names
LeftPath = dataSet.LeftPath
k = dataSet.intrinsicMatrix

cx = k[0,2]
cy = k[1,2]
fx = k[0,0]
fy = k[1,1]

data = np.loadtxt('position_data.csv', delimiter=',')


# Function to filter keypoints based on depth
def filter_keypoints(kp, des, depth_map, min_depth=0, max_depth=40):
    filtered_kp = []
    filtered_des = []
    
    for i, kp_item in enumerate(kp):
        x, y = int(kp_item.pt[0]), int(kp_item.pt[1])
        depth = depth_map[y, x]
        
        if min_depth <= depth <= max_depth:
            filtered_kp.append(kp_item)       # Keep the keypoint
            filtered_des.append(des[i])       # Keep the corresponding descriptor

    filtered_des = np.array(filtered_des) if filtered_des else None
    return filtered_kp, filtered_des

def project_point(pose_params, X):
    R = pose_params[:9].reshape((3, 3))  # First 9 params: rotation matrix (flattened)
    t = pose_params[9:12]  # Next 3 params: translation vector
    
    X_transformed = k@(R @ X + t)  # Apply rotation and then translation

    # Perform the perspective division (assuming intrinsic camera parameters are identity)
    x_projected = X_transformed[:2] / X_transformed[2]  # Normalize by the z-component (perspective projection)
    return x_projected

def reprojection_residuals(params, points_3d, points_2d, visibility, num_frames, num_points):
    residuals = []

    # Reshape params into a list of camera poses (each 12 params: 9 for rotation, 3 for translation)
    poses = params[:num_frames * 12].reshape((num_frames, 12))  # 12 parameters per pose (rotation + translation)
    points_3d_updated = params[num_frames * 12:].reshape((num_points, 3))  # Update the 3D points

    # Compute residuals for each point in each frame
    for j in range(num_points):  # Iterate over points
        for i in range(num_frames):  # Iterate over frames
            pose_params = poses[i]  # Get the 12 parameters (9 for rotation, 3 for translation)
            X = points_3d_updated[j]  # Get the updated 3D point

            # Extract the observed 2D point from the 3D array (621, 3, 2)
            x_observed = points_2d[j, i, :]  # (2,) Extract the 2D point (x, y)

            # Project the 3D point to 2D
            x_projected = project_point(pose_params, X)  # Project the 3D point to 2D

            # Compute the 2D reprojection error
            residuals.append((x_projected - x_observed)*visibility[j,i])

    return np.array(residuals).ravel()


for p in range(1,900,1):
    index = p

    Img1 = cv2.imread(LeftPath + image_names[index], 0)
    Img2 = cv2.imread(LeftPath + image_names[index+1], 0)
    Img3 = cv2.imread(LeftPath + image_names[index+2], 0)
    Img4 = cv2.imread(LeftPath + image_names[index+3], 0)
    Img5 = cv2.imread(LeftPath + image_names[index+4], 0)
    Img6 = cv2.imread(LeftPath + image_names[index+5], 0)
    Img7 = cv2.imread(LeftPath + image_names[index+6], 0)
    Img8 = cv2.imread(LeftPath + image_names[index+7], 0)
    Img9 = cv2.imread(LeftPath + image_names[index+8], 0)
    Img10 = cv2.imread(LeftPath + image_names[index+9], 0)


    orb = cv2.ORB_create()

    # Detect keypoints and descriptors for the images
    kp1, des1 = orb.detectAndCompute(Img1, None)
    kp2, des2 = orb.detectAndCompute(Img2, None)
    kp3, des3 = orb.detectAndCompute(Img3, None)
    kp4, des4 = orb.detectAndCompute(Img4, None)
    kp5, des5 = orb.detectAndCompute(Img5, None)
    kp6, des6 = orb.detectAndCompute(Img6, None)
    kp7, des7 = orb.detectAndCompute(Img7, None)
    kp8, des8 = orb.detectAndCompute(Img8, None)
    kp9, des9 = orb.detectAndCompute(Img9, None)
    kp10, des10 = orb.detectAndCompute(Img10, None)

    # Load depth maps for the images
    depth1 = dataSet.viewDepth(index)
    depth2 = dataSet.viewDepth(index+1)
    depth3 = dataSet.viewDepth(index+2)
    depth4 = dataSet.viewDepth(index+3)
    depth5 = dataSet.viewDepth(index+4)
    depth6 = dataSet.viewDepth(index+5)
    depth7 = dataSet.viewDepth(index+6)
    depth8 = dataSet.viewDepth(index+7)
    depth9 = dataSet.viewDepth(index+8)
    depth10 = dataSet.viewDepth(index+9)


    # Filter the keypoints and descriptors based on depth
    kp1, des1 = filter_keypoints(kp1, des1, depth1)
    kp2, des2 = filter_keypoints(kp2, des2, depth2)
    kp3, des3 = filter_keypoints(kp3, des3, depth3)
    kp4, des4 = filter_keypoints(kp4, des4, depth4)
    kp5, des5 = filter_keypoints(kp5, des5, depth5)
    kp6, des6 = filter_keypoints(kp6, des6, depth6)
    kp7, des7 = filter_keypoints(kp7, des7, depth7)
    kp8, des8 = filter_keypoints(kp8, des8, depth8)
    kp9, des9 = filter_keypoints(kp9, des9, depth9)
    kp10, des10 = filter_keypoints(kp10, des10, depth10)

    # Store keypoints and descriptors in lists
    keypoints = [kp1, kp2, kp3, kp4, kp5, kp6, kp7, kp8, kp9, kp10]
    descriptors = [des1, des2, des3, des4, des5, des6, des7, des8, des9, des10]
    depthValues = [depth1, depth2, depth3, depth4, depth5, depth6, depth7 ,depth8, depth9, depth10]


    # Initialize visibility matrix for keypoints
    visibility =[[],[],[],[],[],[],[],[],[],[]]
    frames = [[],[],[],[],[],[],[],[],[],[]]
    depth_map=[[],[],[],[],[],[],[],[],[],[]]

    for i in range(len(keypoints)):  
        kpo = keypoints[i]   
        deso = descriptors[i] 
        dep = depthValues[i]
        for n, kp_item in enumerate(kpo):
            x, y = int(kp_item.pt[0]), int(kp_item.pt[1])
            depth = dep[y, x]
            X_world = (x - cx) * depth / fx
            Y_world = (y - cy) * depth / fy
            depth_map[i].append([X_world,Y_world,depth])

        depth_map[i] = np.array(depth_map[i]).reshape(len(depth_map[i]),3)    


        # Initialize the Q matrix and array for keypoint coordinates
        Q = np.zeros((len(kpo), len(keypoints)), dtype=int)
        Q[:, i] = 1  # Set first column to 1 (since we are processing image 1)

        arr = np.zeros((len(kpo), len(keypoints), 2), dtype=int)  # Array to store keypoint coordinates

        # Fill the ith column of arr with the coordinates of keypoints from image 1
        for idx, kp in enumerate(kpo):
            x, y = int(kp.pt[0]), int(kp.pt[1])   # Keypoint coordinates
            arr[idx, i, :] = [x, y]    # Store keypoint coordinates in ith column

        
        for j in range(i+1, len(keypoints)): 
            kpn = keypoints[j]  
            desn = descriptors[j] 
 

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            if deso is None or desn is None:
                break

            matches = bf.match(deso, desn)

            matched_indices_img2 = []

            for match in matches:
                idx_img1 = match.queryIdx  
                idx_img2 = match.trainIdx  

                Q[idx_img1, j] = 1
                
                x2, y2 = int(kpn[idx_img2].pt[0]), int(kpn[idx_img2].pt[1])
                
                arr[idx_img1, j, :] = np.array([x2, y2])

                matched_indices_img2.append(idx_img2)


            matched_indices_img2 = list(set(matched_indices_img2)) 
            if len(matched_indices_img2) != 0:
                kpn = [kp for idx, kp in enumerate(kpn) if idx not in matched_indices_img2]
                desn = [des for idx, des in enumerate(desn) if idx not in matched_indices_img2]

                keypoints[j] = kpn
                descriptors[j] = np.array(desn) if len(desn) > 0 else None

        visibility[i] = Q  
        frames[i] = arr

    VisibilityMatrix = np.vstack((visibility[0],visibility[1]))
    Points2D = np.vstack((frames[0],frames[1]))
    Points3D = np.vstack((depth_map[0],depth_map[1]))

    for it in range(2,len(keypoints)):
        VisibilityMatrix = np.vstack((VisibilityMatrix,visibility[it]))
        Points2D = np.vstack((Points2D,frames[it]))
        Points3D = np.vstack((Points3D,depth_map[it]))

    Optdata = np.loadtxt('optimized_data.csv', delimiter=',')
    initial_poses = Optdata[index:index+9,:12]
    initial_poses = np.vstack((initial_poses,data[index+10, :12])) 


    n_points = VisibilityMatrix.shape[0]
    n_frames = len(keypoints)

    initial_params = np.hstack([initial_poses.ravel(), Points3D.ravel()])

    n_residuals = n_frames*n_points*2  # Each point in each frame gives two residuals (x, y)
    n_params = n_frames * 12 + n_points * 3
    sparsity = lil_matrix((n_residuals, n_params), dtype=int)


    for j in range(n_points):
        for i in range(n_frames):
            if VisibilityMatrix[j,i]:
                res_index = 2 * (j * n_frames + i)  # Residual index
                sparsity[res_index, i * 12:(i + 1) * 12] = 1  # Camera pose indices
                sparsity[res_index + 1, i * 12:(i + 1) * 12] = 1
                sparsity[res_index, n_frames * 12 + j] = 1  # 3D point index
                sparsity[res_index + 1, n_frames * 12 + j] = 1


    # Solve the least squares problem using the Trust Region Reflective algorithm with a sparse Jacobian
    result = least_squares(reprojection_residuals, initial_params, args=(Points3D, Points2D, VisibilityMatrix, n_frames, n_points), 
                        method='trf', jac_sparsity=sparsity, verbose=2)

    # Reshape the optimized parameters back into camera poses and 3D points
    optimized_poses = result.x[:n_frames * 12].reshape((n_frames, 12))  # Reshape optimized poses
    optimized_points_3d = result.x[n_frames * 12:].reshape((n_points, 3))  # Reshape optimized 3D points

    i = np.array(optimized_poses[-1])
    rot = i[:9]
    print(i[9:])
    disp = i[9:]
    with open('optimized_data.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(rot.flatten().tolist()+disp.flatten().tolist())


print("--------------------")
print("Done")
print("--------------------")



