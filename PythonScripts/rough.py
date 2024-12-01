import cv2
import numpy as np

path = '/home/shriram/Documents/VisualOdometry/KittiProject/data_odometry_gray/dataset/sequences/00/image_0/'
# Load two consecutive frames (convert them to grayscale)
prev_frame = cv2.imread(path+'000000.png', cv2.IMREAD_GRAYSCALE)
next_frame = cv2.imread(path+'000040.png', cv2.IMREAD_GRAYSCALE)

def adaptive_non_maximum_suppression(kps, top_n=500, min_dist=10):
    """
    Apply Adaptive Non-Maximal Suppression (ANMS) to spread out keypoints.
    kps: List of keypoints
    top_n: Number of keypoints to select
    min_dist: Minimum distance between keypoints after suppression
    """
    # Convert keypoints to a NumPy array of (x, y) positions
    points = np.array([kp.pt for kp in kps])

    # Calculate the distance between every pair of keypoints
    dists = np.linalg.norm(points[:, np.newaxis] - points, axis=2)

    # Compute the response (sharpness) of each keypoint
    responses = np.array([kp.response for kp in kps])

    # Initialize a list to store the selected keypoints
    selected_kps = []

    # Loop through all the keypoints and select the ones based on ANMS
    for _ in range(top_n):
        max_response_index = np.argmax(responses)
        max_response_point = points[max_response_index]

        # Check if the selected point is far enough from the existing points
        if len(selected_kps) == 0 or np.all(np.linalg.norm(np.array(selected_kps) - max_response_point, axis=1) >= min_dist):
            selected_kps.append(max_response_point)
        
        # Suppress the selected point's response for future iterations
        responses[max_response_index] = -1  # Make it a very low response

    # Convert the selected points back to keypoints
    selected_kps = [cv2.KeyPoint(x, y, 1) for x, y in selected_kps]

    return selected_kps


# Initialize the ORB detector
orb = cv2.ORB_create()

# Detect ORB keypoints and descriptors in the previous frame
kp1, des1 = orb.detectAndCompute(prev_frame, None)

# Apply ANMS to get the top 500 spread-out keypoints
top_kps = adaptive_non_maximum_suppression(kp1, top_n=500)

# Convert the keypoints into a format suitable for optical flow (i.e., points)
prevPts = np.float32([kp.pt for kp in kp1]).reshape(-1, 1, 2)
print(prevPts.shape)

# Calculate optical flow using PyrLK
nextPts, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, next_frame, prevPts, None, winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
num_tracked_points = np.sum(status == 1)
print(f"Number of successfully tracked points: {num_tracked_points}")
# Filter the valid points based on status
good_prevPts = prevPts[status == 1]
good_nextPts = nextPts[status == 1]

# Draw the keypoints and flow vectors on the next frame
output = cv2.cvtColor(next_frame, cv2.COLOR_GRAY2BGR)

for i, (new, old) in enumerate(zip(good_nextPts, good_prevPts)):
    x_new, y_new = new.ravel()
    x_old, y_old = old.ravel()
    output = cv2.circle(output, (int(x_new), int(y_new)), 5, (0, 255, 0), -1)  # Draw the point
    output = cv2.line(output, (int(x_new), int(y_new)), (int(x_old), int(y_old)), (0, 0, 255), 2)  # Draw the flow line

# Show the result
cv2.imshow('Optical Flow with ANMS and ORB', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
