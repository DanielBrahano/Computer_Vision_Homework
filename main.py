import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
from extract_info_from_puzzles import *
from output_results import *


def get_sift_detector_descriptor(image):
    # Create a SIFT object
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)

    return keypoints, descriptors


def compute_euclidean_distance(descriptor1, descriptor2):
    if descriptor1 is None or descriptor2 is None:
        return None
    N = (len(descriptor1))
    M = len(descriptor2)

    dist_matrix = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            dist_matrix[i][j] = np.linalg.norm(descriptor1[i] - descriptor2[j])
    return dist_matrix


def sift_ratio_match(ratio_thresh, dist_matrix, min_matches):
    ''' Extracting good matches, function returns (i, idx1)
    return values:
    i - is the index of the keypoint in the first image
    idx -  the index of the matching keypoint in the second image '''
    # Apply ratio test to keep only the good matches
    good_matches = []
    position = []
    for i in range(len(dist_matrix)):
        idx1 = np.argmin(dist_matrix[i])
        dists = dist_matrix[i].copy()
        dists[idx1] = np.inf
        idx2 = np.argmin(dists)
        ratio = dist_matrix[i][idx1] / dist_matrix[i][idx2]
        if ratio < ratio_thresh:
            match = cv2.DMatch(i, idx1, dist_matrix[i][idx1])
            good_matches.append(match)
            position.append((i, idx1))

    print('len matches list:', str(len(good_matches)))
    if len(good_matches) < min_matches:
        return None

    return good_matches


def myPerspectiveTransform(perspectiveMatrix, sourcePoints):
    """
    Applies a perspective transformation to a set of 2D points.

    Args:
        perspectiveMatrix: A 3x3 matrix representing the perspective transformation.
        sourcePoints: An Nx2 numpy array containing the coordinates of the points to transform.

    Returns:
        An Nx2 array containing the transformed points.
    """

    # Flatten the input points and convert to homogeneous coordinates
    sourcePoints = sourcePoints.reshape((-1, 2))

    # Add a column of ones to the points to allow for translation
    sourcePoints_h = np.hstack((sourcePoints, np.ones((len(sourcePoints), 1))))

    # Apply the perspective transformation to the points
    destPoints_h = perspectiveMatrix.dot(sourcePoints_h.T).T

    # Convert back to non-homogeneous coordinates
    destPoints = destPoints_h[:, :2] / destPoints_h[:, 2].reshape((-1, 1))

    # Return the transformed points
    return destPoints.reshape(-1, 1, 2)


def show_keypoints(image1, keypoints1, image2, keypoints2):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(16, 9))
    ax1.imshow(cv2.drawKeypoints(image1, keypoints1, None, color=(0, 255, 0)))
    ax1.set_xlabel('(a)')

    ax2.imshow(cv2.drawKeypoints(image2, keypoints2, None, color=(0, 255, 0)))
    ax2.set_xlabel('(b)')


def residual_calculate(transformed_points, dst_points, e):
    '''
    add the pair if distance less than error allowed
    '''
    residuals = np.linalg.norm(transformed_points - dst_points, axis=2).flatten()
    inliers_index = np.where(residuals < e)
    final_inliers = np.concatenate((transformed_points[inliers_index], dst_points[inliers_index]), axis=1)
    return final_inliers


def is_nice_homography(H):
    if len(H) == 0:
        return False

    det = H[0, 0] * H[1, 1] - H[1, 0] * H[0, 1];
    if det < 0.0:
        return False

    N1 = math.sqrt(H[0, 0] * H[0, 0] + H[1, 0] * H[1, 0]);
    if N1 > 4 or N1 < 0.1:
        return False

    N2 = math.sqrt(H[0, 1] * H[0, 1] + H[1, 1] * H[1, 1]);
    if N2 > 4 or N2 < 0.1:
        return False

    N3 = math.sqrt(H[2, 0] * H[2, 0] + H[2, 1] * H[2, 1]);
    if N3 > 0.002:
        return False

    return True


def run_ransac_homography(img1, img2, iterations=10000, e=4, threshold=0.8, min_matches=35):
    kp1, desc1 = get_sift_detector_descriptor(img1)
    kp2, desc2 = get_sift_detector_descriptor(img2)

    # get the distance MXN matrix between every descriptors pair
    distanceOfDescriptors = compute_euclidean_distance(desc1, desc2)
    if distanceOfDescriptors is None:
        return None

    # find matches with ratio test
    matches = sift_ratio_match(0.8, distanceOfDescriptors, min_matches)

    if matches is None:
        return None

    # extract source and destination points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    max_inliers = []
    out_homography_matrix = None
    # ransac loop
    for i in range(iterations):
        # choose 4 random indices
        rnd_4_indeces = random.sample(range(0, len(src_pts)), 4)

        # calculate the transformation for these indices from view1 to view2
        rnd_pairs1 = src_pts[rnd_4_indeces]
        rnd_pairs2 = dst_pts[rnd_4_indeces]

        # get perspective matrix
        homography_matrix = cv2.getPerspectiveTransform(rnd_pairs1,
                                                        rnd_pairs2)

        is_correct_homography = True
        is_correct_homography = is_nice_homography(homography_matrix)
        if is_correct_homography:
            predicted_src_points = myPerspectiveTransform(homography_matrix, src_pts)
            # calculate residuals
            inliers = residual_calculate(predicted_src_points, dst_pts, e)
            if len(inliers) > len(max_inliers):
                max_inliers = inliers
                out_homography_matrix = homography_matrix

        # check if we have enough inliers
        if len(max_inliers) > (len(src_pts) * threshold):
            break

    return out_homography_matrix


def run_ransac_affine(img1, img2, iterations=10000, e=4, threshold=0.8, min_matches=35):
    # get keypoints and descriptors
    kp1, desc1 = get_sift_detector_descriptor(img1)
    kp2, desc2 = get_sift_detector_descriptor(img2)

    #

    # get the distance MXN matrix between every descriptors pair
    distanceOfDescriptors = compute_euclidean_distance(desc1, desc2)
    if distanceOfDescriptors is None:
        return None
    # find matches with ratio test
    matches = sift_ratio_match(threshold, distanceOfDescriptors, min_matches)

    if matches is None:
        return None

    # extract source and destination points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    max_inliers = []
    out_affine_matrix = None
    # ransac loop
    for i in range(iterations):
        # choose 3 random indices
        rnd_3_indices = random.sample(range(0, len(src_pts)), 3)

        # calculate the transformation for these indices from src to dst
        rnd_pairs1 = src_pts[rnd_3_indices]
        rnd_pairs2 = dst_pts[rnd_3_indices]

        # get affine matrix
        temp_affine_matrix = cv2.getAffineTransform(rnd_pairs1,
                                                    rnd_pairs2)  # or use getPerspectiveTransform from utils.py

        # make it 3x3
        affine_matrix = np.zeros((3, 3))
        for i in range(2):
            for j in range(3):
                affine_matrix[i][j] = temp_affine_matrix[i][j]
        affine_matrix[2][2] = 1

        # calculate residuals
        predicted_src_points = myPerspectiveTransform(affine_matrix, src_pts)
        inliers = residual_calculate(predicted_src_points, dst_pts, e)
        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            out_affine_matrix = temp_affine_matrix

        # check if we have enough inliers
        if len(max_inliers) > (len(src_pts) * threshold):
            break

    return out_affine_matrix


def wrap_affine(img1, img2, height, width, transformation_matrix):
    img_output = cv2.warpAffine(img1, transformation_matrix, (width, height), flags=cv2.INTER_CUBIC)
    relative = img_output.copy()
    grayImg = cv2.cvtColor(img_output, cv2.COLOR_BGR2GRAY)
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if grayImg[i][j] == 0:
                img_output[i][j] = img2[i][j]
    return img_output, relative


def wrap_homography(img1, img2, height, width, transformation_matrix):
    img_output = cv2.warpPerspective(img1, transformation_matrix, (width, height), flags=cv2.INTER_CUBIC)
    relative = img_output.copy()
    grayImg = cv2.cvtColor(img_output, cv2.COLOR_BGR2GRAY)
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if grayImg[i][j] == 0:
                img_output[i][j] = img2[i][j]
    # save relative image
    return img_output, relative


# get all affine puzzles and info
all_affine_puzzle = get_affine_images_and_all_info()
num_of_piece_for_each_affine_puzzle = []

# for i in range(len(all_affine_puzzle)):
for i in range(7, 10):

    if i == 7:
        minimum_matches=8
    elif i == 8:
        minimum_matches = 17
    else:
        minimum_matches = 35


    affine_puzzle_i, height, width, final_warp_mat = all_affine_puzzle[i]
    # how many iterations have been with no matches
    no_matches_in_puzzle = 0
    # num of matches so far
    nMatches = 0
    j = 0
    num_iterations = 0
    # pieces assembled list
    matches_list = []
    imgOutput_affine = affine_puzzle_i[j]
    imgOutput_affine = cv2.warpAffine(imgOutput_affine, final_warp_mat, (width, height), flags=cv2.INTER_CUBIC)
    while nMatches < len(affine_puzzle_i):
        if j not in matches_list:
            # convert to gray scale
            affine_puzzle_i_j_gray = cv2.cvtColor(affine_puzzle_i[j], cv2.COLOR_BGR2GRAY)
            imgOutput_affine_gray = cv2.cvtColor(imgOutput_affine, cv2.COLOR_BGR2GRAY)
            h = run_ransac_affine(affine_puzzle_i_j_gray, imgOutput_affine_gray, min_matches=minimum_matches)
            if h is not None:
                imgOutput_affine, relative = wrap_affine(affine_puzzle_i[j], imgOutput_affine, height, width, h)
                save_relative_image(relative, 'homography', i + 1, j + 1)
                nMatches = nMatches + 1
                matches_list.append(j)
                no_matches_in_puzzle = -1

        no_matches_in_puzzle += 1
        j = j + 1
        j = j % len(affine_puzzle_i)
        num_iterations = num_iterations + 1
        print("num_iterations = " + str(num_iterations))
        print('matches: ')
        print(matches_list)
        print("nMatches = " + str(nMatches))
        if num_iterations > min((len(affine_puzzle_i)) ** 2, (len(affine_puzzle_i)) ** 2):
            break
        if no_matches_in_puzzle > 2 * len(affine_puzzle_i):
            break

    num_of_piece_for_each_affine_puzzle.append((i + 1, nMatches))
    save_solution_image(imgOutput_affine, 'affine', i+1, nMatches, len(affine_puzzle_i))

    figs = plt.figure(figsize=(8, 8))
    plt.imshow(imgOutput_affine)

all_homography_puzzle = get_homography_images_and_all_info()
num_of_piece_for_each_homography_puzzle = []

# loop over all homography puzzles and assemble
# for i in range(10, 10):
for i in range(0, len(all_homography_puzzle)):
#for i in range(0, 2):

    if i == 5 or i == 6 or i == 7:
        minimum_matches=10
    else:
        minimum_matches=35

    homography_puzzle_i, height, width, final_warp_mat = all_homography_puzzle[i]

    # how many iterations have been with no matches
    no_matches_in_puzzle = 0
    # num of matches so far
    nMatches = 0
    j = 0
    num_iterations = 0
    # pieces assembled list
    matches_list = []
    imgOutput_homography = homography_puzzle_i[j]
    imgOutput_homography = cv2.warpPerspective(imgOutput_homography, final_warp_mat, (width, height),
                                               flags=cv2.INTER_CUBIC)
    while nMatches < len(homography_puzzle_i):
        if j not in matches_list:
            # convert to gray scale
            homography_puzzle_i_j_gray = cv2.cvtColor(homography_puzzle_i[j], cv2.COLOR_BGR2GRAY)
            imgOutput_homography_gray = cv2.cvtColor(imgOutput_homography, cv2.COLOR_BGR2GRAY)
            h = run_ransac_homography(homography_puzzle_i_j_gray, imgOutput_homography_gray,
                                      min_matches=minimum_matches)
            if h is not None:
                imgOutput_homography, relative = wrap_homography(homography_puzzle_i[j], imgOutput_homography, height,
                                                                 width, h)
                #save_relative_image(relative, 'homography', i + 1, j + 1)
                no_matches_in_puzzle = -1
                nMatches = nMatches + 1
                matches_list.append(j)

        no_matches_in_puzzle += 1
        j = j + 1
        j = j % len(homography_puzzle_i)
        num_iterations = num_iterations + 1
        print("num_iterations = " + str(num_iterations))
        print('matches: ')
        print(matches_list)
        print("nMatches = " + str(nMatches))
        if num_iterations > (len(homography_puzzle_i)) ** 2:
            break

        if no_matches_in_puzzle > 2 * len(homography_puzzle_i):
            break

    num_of_piece_for_each_homography_puzzle.append((i + 1, nMatches))
    #save_solution_image(imgOutput_homography, 'homography', i+1, nMatches, len(homography_puzzle_i))

    figs = plt.figure(figsize=(8, 8))
    plt.imshow(imgOutput_homography)

print('##################### RESULTS #####################')
# print results for affine puzzles
for puzzle, nMatches in num_of_piece_for_each_affine_puzzle:
    print('for affine puzzle', puzzle,
          'we assembled ' + str(nMatches) + '/' + str(num_of_pieces_in_puzzle_affine[puzzle - 1]) + ' pieces')

# print results for homography puzzles
for puzzle, nMatches in num_of_piece_for_each_homography_puzzle:
    print('for homography puzzle', puzzle,
          'we assembled ' + str(nMatches) + '/' + str(num_of_pieces_in_puzzle_homography[puzzle - 1]) + ' pieces')
plt.show()
