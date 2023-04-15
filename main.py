import cv2
import matplotlib.pyplot as plt
import numpy as np
import cv2
import numpy as np

def ransac_affine(img1, img2, num_iterations=1000, inlier_threshold=10):
    # Detect keypoints and compute descriptors using SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Match keypoints using FLANN matcher
    flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})
    matches = flann.knnMatch(des1, des2, k=2)

    # Extract matched keypoints
    pts1 = []
    pts2 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.float32(pts1).reshape(-1, 1, 2)
    pts2 = np.float32(pts2).reshape(-1, 1, 2)

    # Run RANSAC to estimate affine transformation
    best_inliers = []
    for i in range(num_iterations):
        # Randomly select 3 keypoints
        indices = np.random.choice(len(pts1), size=3, replace=False)
        src_pts = pts1[indices]
        dst_pts = pts2[indices]

        # Compute affine transformation matrix
        M, mask = cv2.estimateAffine2D(src_pts, dst_pts)

        if M is not None:
            # Calculate residuals
            dst_pts_pred = cv2.transform(src_pts, M)
            residuals = np.sqrt(np.sum(np.square(dst_pts - dst_pts_pred), axis=2))

            # Count inliers and update best set of inliers
            inliers = np.sum(residuals < inlier_threshold)
            if inliers > len(best_inliers):
                best_inliers = np.where(residuals < inlier_threshold)[0]

    # Refit affine transformation on best set of inliers
    if len(best_inliers) > 0:
        src_pts = pts1[best_inliers]
        dst_pts = pts2[best_inliers]
        M, _ = cv2.estimateAffine2D(src_pts, dst_pts)

        # Warp second image to align with first image
        img2_warped = cv2.warpAffine(img2, M, (img1.shape[1], img1.shape[0]))

        # Return warped image and affine transformation matrix
        return img2_warped, M

    # Return None if no inliers found
    return None, None


def print_hi(name):
    # Use a breakpopip install opencv-python==3.3.0.10 opencv-contrib-python==3.3.0.10int in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img1 = cv2.imread('puzzles/puzzle_affine_4/pieces/piece_4.jpg')
    img2 = cv2.imread('puzzles/puzzle_affine_4/pieces/piece_4.jpg')

    # View the images
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(16, 9))
    ax1.imshow(img1, cmap='gray')
    ax1.set_xlabel('img1', fontsize=14)

    ax2.imshow(img2, cmap='gray')
    ax2.set_xlabel('img2', fontsize=14)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Check if kp1 or des1 is None
    if kp1 is None or des1 is None:
        print("SIFT failed to detect keypoints or compute descriptors for img1")
    else:
        # Print the number of keypoints and descriptors for img1
        print("Number of keypoints in img1: ", len(kp1))
        print("Number of descriptors in img1: ", des1.shape[0])

    # Check if kp2 or des2 is None
    if kp2 is None or des2 is None:
        print("SIFT failed to detect keypoints or compute descriptors for img2")
    else:
        # Print the number of keypoints and descriptors for img2
        print("Number of keypoints in img2: ", len(kp2))
        print("Number of descriptors in img2: ", des2.shape[0])



    plt.show()

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
