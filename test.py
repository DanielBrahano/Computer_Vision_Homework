import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

'''Get keypoints and descriptor of an image'''


def get_sift_detector_descriptor(image):
    # Create a SIFT object
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)

    return keypoints, descriptors


def show_keypoints(image1, keypoints1, image2, keypoints2):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(28, 0))
    ax1.imshow(cv2.drawKeypoints(image1, keypoints1, None, color=(0, 255, 0)))
    ax1.set_xlabel('(a)')

    ax2.imshow(cv2.drawKeypoints(image2, keypoints2, None, color=(0, 255, 0)))
    ax2.set_xlabel('(b)')


def create_matching_object(crossCheck):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    return bf


def keypoints_matching(feature_train_img, feature_query_img):
    bf = create_matching_object(crossCheck=True)

    best_matches = bf.match(feature_train_img, feature_query_img)

    raw_matches = sorted(best_matches, key=lambda x: x.distance)
    print('Raw matches with Brute Force', len(raw_matches))

    return raw_matches


def keypoints_matching_KNN(feature_train_img, feature_query_img, ratio):
    bf = create_matching_object(crossCheck=False)

    raw_matches = bf.knnMatch(feature_train_img, feature_query_img, k=2)
    print('Raw matches with KNN', len(raw_matches))

    knn_matches = []

    for m, n in raw_matches:
        if m.distance < n.distance * ratio:
            knn_matches.append(m)

    return raw_matches


def compute_euclidean_distance(descriptor1, descriptor2):
    N = (len(descriptor1))
    M = len(descriptor2)

    dist_matrix = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            dist_matrix[i, j] = np.linalg.norm(descriptor1[i] - descriptor2[j])
    return dist_matrix


''' Extracting good matches, function returns tuple (i, idx1)
function calculates the ratio between (f_i - f'_j*) and (f_i - f'_j**) (lec 3 presentation slide 88/93)
return values:
i - is the index of the keypoint in the first image
idx -  the index of the matching keypoint in the second image '''


def sift_ratio_match(ratio_thresh, dist_matrix):
    # Apply ratio test to keep only the good matches
    good_matches = []
    for i in range(len(dist_matrix)):
        idx1 = np.argmin(dist_matrix[i])
        dists = dist_matrix[i].copy()
        dists[idx1] = np.inf
        idx2 = np.argmin(dists)
        ratio = dist_matrix[i][idx1] / dist_matrix[i][idx2]
        if ratio < ratio_thresh:
            match = cv2.DMatch(i, idx1, dist_matrix[i][idx1])
            good_matches.append(match)

    return good_matches


def homography_stiching(keypoints_train_img, keypoints_query_img, matches, reprojTresh):
    # convert to numpy array
    keypoints_train_img = np.float32([keypoint.pt for keypoint in keypoints_train_img])
    keypoints_query_img = np.float32([keypoint.pt for keypoint in keypoints_query_img])

    if len(matches) > 4:
        points_train = np.float32([keypoints_train_img[m.queryIdx] for m in matches])
        points_query = np.float32([keypoints_query_img[m.trainIdx] for m in matches])

        (H, status) = cv2.findHomography(points_train, points_query, cv2.RANSAC, reprojTresh)

        return matches, H, status
    else:
        return None


def affine_stiching(keypoints_train_img, keypoints_query_img, matches, reprojTresh):
    # convert to numpy array
    keypoints_train_img = np.float32([keypoint.pt for keypoint in keypoints_train_img])
    keypoints_query_img = np.float32([keypoint.pt for keypoint in keypoints_query_img])

    status = True
    if len(matches) > 3:
        points_train = np.float32([keypoints_train_img[m.queryIdx] for m in matches])
        points_query = np.float32([keypoints_query_img[m.trainIdx] for m in matches])

        input_pts = random.sample(list(points_train), 3)
        output_pts = random.sample(list(points_query), 3)

        input_pts = np.float32(input_pts)
        output_pts = np.float32(output_pts)

        A = cv2.getAffineTransform(input_pts, output_pts)

        return matches, A, status
    else:
        return None


def affine_solver(keypoints_train_img, keypoints_query_img, matches):
    # convert to numpy array
    keypoints_train_img = np.float32([keypoint.pt for keypoint in keypoints_train_img])
    keypoints_query_img = np.float32([keypoint.pt for keypoint in keypoints_query_img])

    status = True
    if len(matches) > 3:
        points_train = np.float32([keypoints_train_img[m.queryIdx] for m in matches])
        points_query = np.float32([keypoints_query_img[m.trainIdx] for m in matches])

        input_pts = random.sample(list(points_train), 3)
        output_pts = random.sample(list(points_query), 3)

        input_pts = np.float32(input_pts)
        output_pts = np.float32(output_pts)

        A = cv2.getAffineTransform(input_pts, output_pts)

        return matches, A, status
    else:
        return None


features_to_match = 'mine'

path = 'puzzles/puzzle_affine_1/pieces/piece_1.jpg'
path2 = 'puzzles/puzzle_affine_1/pieces/piece_2.jpg'

train_image = cv2.imread(path)
# Convert the image to grayscale
train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)
train_image_gray = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY)

# Do the same for query image
query_image = cv2.imread(path2)
# Convert the image to grayscale
query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
query_image_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)

# View the images
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(16, 9))
ax1.imshow(query_image, cmap='gray')
ax1.set_xlabel('Query Image', fontsize=14)

ax2.imshow(train_image, cmap='gray')
ax2.set_xlabel('Train Image(Photo to Transform)', fontsize=14)

''' STEP 1'''
# get keypoints and decriptors
train_image_keypoints, train_image_descriptors = get_sift_detector_descriptor(train_image_gray)
query_image_keypoints, query_image_descriptors = get_sift_detector_descriptor(query_image_gray)

# test print
print('train_image_keypoints', train_image_keypoints)
print('train_image_descriptors', train_image_descriptors)

# print keyoints
show_keypoints(train_image, train_image_keypoints, query_image, query_image_keypoints)

# draw matched features
print('Drawing matched features for', features_to_match)

figs = plt.figure(figsize=(20, 8))

if features_to_match == 'bf':
    matches = keypoints_matching(train_image_descriptors, query_image_descriptors)
    # draw the matches between images
    mapped_feature_image = cv2.drawMatches(train_image, train_image_keypoints, query_image, query_image_keypoints,
                                           matches[:300], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(mapped_feature_image)

elif features_to_match == 'knn':
    matches = keypoints_matching_KNN(train_image_descriptors, query_image_descriptors, ratio=0.75)
    mapped_feature_image = cv2.drawMatchesKnn(train_image, train_image_keypoints, query_image, query_image_keypoints,
                                              matches[0:30], None, flags=2)
    plt.imshow(mapped_feature_image)

    '''STEP 2'''
    '''STEP 3'''
elif features_to_match == 'mine':
    distance_matrix = compute_euclidean_distance(train_image_descriptors, query_image_descriptors)
    matches = sift_ratio_match(0.8, distance_matrix)
    mapped_feature_image = cv2.drawMatches(train_image, train_image_keypoints, query_image, query_image_keypoints,
                                           matches[:300], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(mapped_feature_image)

# M = homography_stiching(train_image_keypoints, query_image_keypoints, matches, reprojTresh=4)
#M = affine_stiching(train_image_keypoints, query_image_keypoints, matches, reprojTresh=4)
M = affine_solver(train_image_keypoints, query_image_keypoints, matches)

if M is None:
    print('Error')
(matches, Homography_Matrix, status) = M

width = query_image.shape[1] + train_image.shape[1]
height = query_image.shape[0] + train_image.shape[0]

# result = cv2.warpPerspective(train_image, Homography_Matrix, (width, height))
result = cv2.warpAffine(train_image, Homography_Matrix, (width, height))

result[0:query_image.shape[0], 0:query_image.shape[1]] = query_image

plt.figure(figsize=(20, 10))
plt.axis('off')
plt.imshow(result)
plt.show()
