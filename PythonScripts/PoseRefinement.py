import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

k=np.eye(3)
# Define the projection of 3D points to 2D given a rotation matrix (flattened 9 params) and translation vector (3 params)
def project_point(pose_params, X):
    R = pose_params[:9].reshape((3, 3))  # First 9 params: rotation matrix (flattened)
    t = pose_params[9:12]  # Next 3 params: translation vector
    
    X_transformed = k@(R @ X + t)  # Apply rotation and then translation

    # Perform the perspective division (assuming intrinsic camera parameters are identity)
    x_projected = X_transformed[:2] / X_transformed[2]  # Normalize by the z-component (perspective projection)
    return x_projected

# Residual function to compute reprojection error
def reprojection_residuals(params, points_3d, points_2d, visibility, num_frames, num_points):
    residuals = []

    # Reshape params into a list of camera poses (each 12 params: 9 for rotation, 3 for translation) and the 3D points
    poses = params[:num_frames * 12].reshape((num_frames, 12))  # 12 parameters per pose (rotation + translation)
    points_3d_updated = params[num_frames * 12:].reshape((num_points, 3))  # Update the 3D points

    # Compute residuals for each point in each frame
    for j in range(num_points):  # Iterate over points
        for i in range(num_frames):  # Iterate over frames
            if visibility[j, i]:  # Only compute residual if point is visible in the frame
                pose_params = poses[i]  # Get the 12 parameters (9 for rotation, 3 for translation)
                X = points_3d_updated[j]  # Get the updated 3D point

                # Calculate the correct index for the 2D point in the flattened array
                index_2d = 2 * (j * num_frames + i)  # Index in the flattened 2D array
                x_observed = points_2d[index_2d:index_2d + 2]  # Extract the 2D point (x, y)

                x_projected = project_point(pose_params, X)  # Project the 3D point to 2D
                residuals.append(x_projected - x_observed)  # Compute the 2D reprojection error

    return np.array(residuals).ravel()  # Flatten residuals

# Generate random 3D points (n_points points in 3D space)
n_points = 621
points_3d = np.random.rand(n_points, 3) * 100  # Random points in [0, 100] range

# Generate random 2D projections in n_frames (flattened 2D points array)
n_frames = 3
points_2d = np.random.rand(n_points * n_frames * 2) * 1000  # Random 2D points in [0, 1000] range

print(points_2d.shape)

# Visibility matrix (n_points x n_frames) indicating whether each point is visible in each frame
visibility = np.ones((n_points, n_frames), dtype=bool)  # Assume all points are visible in all frames

# Initial guess for camera poses (n_frames poses x 12 params: 9 for rotation, 3 for translation)
initial_poses = np.random.rand(n_frames, 12)

# Flatten the initial camera pose parameters and add the 3D points as well
initial_params = np.hstack([initial_poses.ravel(), points_3d.ravel()])

# Define the sparsity pattern of the Jacobian matrix
n_residuals = n_frames * n_points * 2  # Each point in each frame gives two residuals (x, y)
n_params = n_frames * 12 + n_points * 3  # 12 parameters per pose, 3 parameters per 3D point

sparsity = lil_matrix((n_residuals, n_params), dtype=int)

# Fill the sparsity matrix
for j in range(n_points):
    for i in range(n_frames):
        if visibility[j, i]:  # Only add entries for visible points
            # Calculate the correct index for residuals associated with this point in this frame
            res_index = 2 * (j * n_frames + i)
            
            # Residuals associated with the camera pose (12 parameters: 9 for rotation, 3 for translation)
            sparsity[res_index, i * 12:(i + 1) * 12] = 1
            sparsity[res_index + 1, i * 12:(i + 1) * 12] = 1
            
            # Residuals associated with the 3D point (3 parameters)
            sparsity[res_index, n_frames * 12 + j * 3:n_frames * 12 + (j + 1) * 3] = 1
            sparsity[res_index + 1, n_frames * 12 + j * 3:n_frames * 12 + (j + 1) * 3] = 1

# Solve the least squares problem using the Trust Region Reflective algorithm with a sparse Jacobian
result = least_squares(reprojection_residuals, initial_params, args=(points_3d, points_2d, visibility, n_frames, n_points), 
                       method='trf', jac_sparsity=sparsity, verbose=2)

# Reshape the optimized parameters back into camera poses and 3D points
optimized_poses = result.x[:n_frames * 12].reshape((n_frames, 12))  # Reshape optimized poses
optimized_points_3d = result.x[n_frames * 12:].reshape((n_points, 3))  # Reshape optimized 3D points

# Output the optimized camera poses and 3D points
print("Optimized camera poses (rotation and translation):\n", optimized_poses)
print("Optimized 3D points:\n", optimized_points_3d)

# Check if the optimization was successful
if result.success:
    print("Optimization was successful!")
else:
    print("Optimization failed:", result.message)
