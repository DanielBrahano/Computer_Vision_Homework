import sys
import time as t
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from load_and_save_data import *

'''
---------------------------------COST VOLUME CALCULATION---------------------------------------
'''


def hamming_distance(arr1, arr2):
    return np.sum(arr1 != arr2)


def calculate_volume_cost(
        census_image_left: np.ndarray,
        census_image_right: np.ndarray,
        max_disparity: int,
        height=3,
        width=3
) -> (np.ndarray, np.ndarray):
    # Padding the images
    padding = max_disparity + max(height, width) // 2
    census_image_left = np.pad(census_image_left, padding, mode='constant')
    census_image_right = np.pad(census_image_right, padding, mode='constant')

    h, w = census_image_left.shape  # shape of images

    # initialize cost volumes
    cost_l = np.zeros((h, w, max_disparity), dtype=np.float32)
    cost_r = np.zeros((h, w, max_disparity), dtype=np.float32)

    for d in range(max_disparity):
        for y in range(padding, h - padding):
            for x in range(padding, w - padding):
                cost_l[y, x, d] = hamming_distance(census_image_left[y, x], census_image_right[y, x - d])
                cost_r[y, x, d] = hamming_distance(census_image_right[y, x], census_image_left[y, x + d])

    # Remove padding from the cost volumes
    cost_l = cost_l[padding:-padding, padding:-padding]
    cost_r = cost_r[padding:-padding, padding:-padding]

    return cost_l, cost_r


'''
-----------------------------------CENSUS TRANSFORMATION-----------------------------------------------
'''


def normalize_data(input_data: np.ndarray) -> np.ndarray:
    """ Normalize image to values between 0 and 1. """
    min_val = np.min(input_data)
    max_val = np.max(input_data)
    normalized_data = (input_data - min_val) / (max_val - min_val) if max_val != min_val else input_data

    # ensure values stay within 0 and 1
    normalized_data[normalized_data < 0] = 0
    normalized_data[normalized_data > 1] = 1

    return normalized_data


def calculate_census(input_block: np.ndarray = None) -> int:
    """ Compute census in a numpy-vectorized fashion. """
    if len(input_block.shape) != 2:
        raise Exception('Data must be 2-dimensional')

    # binary census vector
    bin_vector = np.array(input_block < 0).flatten()
    # remove central value
    center_index = (input_block.shape[0] * input_block.shape[1]) // 2
    bin_vector = np.delete(bin_vector, center_index)
    # convert binary vector to integer
    census_num = bin_vector.dot(1 << np.arange(bin_vector.size)[::-1])

    return census_num


def calculate_census(block):
    center_value = block[len(block) // 2, len(block[0]) // 2]
    return block < center_value


def apply_census_transform(input_image_l: np.ndarray, input_image_r: np.ndarray, h, w) -> (np.ndarray, np.ndarray):
    """ Census feature extraction. """

    height, width = input_image_l.shape

    # convert to float and normalize
    input_image_l = normalize_data(input_image_l.astype(float))
    input_image_r = normalize_data(input_image_r.astype(float))

    left_census_values = np.empty(shape=(height, width), dtype=object)
    right_census_values = np.empty(shape=(height, width), dtype=object)


    # Exclude pixels on the border (they will have no census values)
    for y in range(h, height - h):
        for x in range(w, width - w):
            # extract block region from left image and compute its census value
            block_l = np.subtract(
                input_image_l[y - h:y + h + 1, x - w:x + w + 1],
                input_image_l[y, x], dtype=np.float64)
            left_census_values[y, x] = calculate_census(block_l)

            # extract block region from right image and compute its census value
            block_r = np.subtract(
                input_image_r[y - h:y + h + 1, x - w:x + w + 1],
                input_image_r[y, x], dtype=np.float64)
            right_census_values[y, x] = calculate_census(block_r)


    return left_census_values, right_census_values


"""------------------------------LEFT-RIGHT CONSISTENCY TEST---------------------------------
"""


def left_right_consistency_test(disp_map_left, disp_map_right, threshold=0):
    # Initialize two empty consistency maps
    height, width = disp_map_left.shape
    consistency_map_left = np.zeros((height, width), dtype=np.float32)
    consistency_map_right = np.zeros((height, width), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            # Check for out of bounds
            if x - int(disp_map_left[y, x]) < 0 or x + int(disp_map_right[y, x]) >= width:
                continue
            # Perform the consistency check for the left disparity map
            if abs(disp_map_left[y, x] - disp_map_right[y, x - int(disp_map_left[y, x])]) > threshold:
                consistency_map_left[y, x] = 1.0  # Mark as inconsistent in the left consistency map
            # Perform the consistency check for the right disparity map
            if abs(disp_map_right[y, x] - disp_map_left[y, x + int(disp_map_right[y, x])]) > threshold:
                consistency_map_right[y, x] = 1.0  # Mark as inconsistent in the right consistency map

    # Fix disparity maps by ignoring outliers
    fixed_disp_map_left = np.where(consistency_map_left == 0, disp_map_left, 0)
    fixed_disp_map_right = np.where(consistency_map_right == 0, disp_map_right, 0)

    return fixed_disp_map_left, fixed_disp_map_right


'''
------------------------------COST AGGREGATION--------------------------------------------
'''


def cost_aggregation(cost_volume_left, cost_volume_right, kernel_size=15, sigma=1.5):
    # Apply Gaussian blur for local aggregation
    cost_volume_left_agg = cv2.GaussianBlur(cost_volume_left, (kernel_size, kernel_size), 0)
    cost_volume_right_agg = cv2.GaussianBlur(cost_volume_right, (kernel_size, kernel_size), 0)

    # cost_volume_left_agg = cv2.medianBlur(cost_volume_left, kernel_size)
    # cost_volume_right_agg = cv2.medianBlur(cost_volume_right, kernel_size)
    return cost_volume_left_agg, cost_volume_right_agg


'''
------------------------------CALCULATE DEPTH--------------------------------------------
'''


def calculate_depth(disparity_map, baseline_dist, focal_length):
    # Create a mask to handle zeros in the disparity map
    mask = (disparity_map != 0)

    # Calculate depth values with the zero check
    depth_map = np.zeros_like(disparity_map, dtype=np.float32)
    depth_map[mask] = (focal_length * baseline_dist) / disparity_map[mask]
    return depth_map

'''
-----------------------------GET DISPARITY--------------------------------------------
'''


def stereo_algorithm(im_left, im_right, max_disparity, height, width):
    # Convert the color images to grayscale
    gray_left = cv2.cvtColor(im_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(im_right, cv2.COLOR_BGR2GRAY)

    gray_left, gray_right = apply_census_transform(gray_left, gray_right, height, width)

    cost_left, cost_right = calculate_volume_cost(gray_left, gray_right, max_disparity, height, width)

    cost_left, cost_right = cost_aggregation(cost_left, cost_right)

    # Winner takes all
    disp_map_left = np.argmin(cost_left, axis=2)
    disp_map_right = np.argmin(cost_right, axis=2)

    disp_map_left, disp_map_right = left_right_consistency_test(disp_map_left, disp_map_right)

    return disp_map_left, disp_map_right


# TODO: Hana added
def reproject_points(disparity_map, baseline_dist, focal_length, intrinsics):
    """Reproject the image coordinates into 3D space."""
    depth_map = calculate_depth(disparity_map, baseline_dist, focal_length)
    height, width = disparity_map.shape
    points_3d = np.zeros((height, width, 3), dtype=np.float32)

    # TODO: Hana: I am not sure if we need to inverse or if K is already inverse.
    K_inv = np.linalg.inv(intrinsics)

    for y in range(height):
        for x in range(width):
            depth = depth_map[y, x]
            homogeneous_coords = np.array([x, y, 1])
            camera_coords = depth * np.dot(K_inv, homogeneous_coords)
            points_3d[y, x] = camera_coords

    return points_3d


def project_points(points_3d, intrinsics):
    """Return the reprojected points to the original camera plane."""

    # TODO: Hana: I am not sure if we need to inverse or if K is already inverse.
    K_inv = np.linalg.inv(intrinsics)

    height, width, _ = points_3d.shape
    points_3d_reshaped = points_3d.reshape((height * width, 3)).T

    points_2d_homogeneous = K_inv.dot(points_3d_reshaped)
    points_2d = (points_2d_homogeneous / points_2d_homogeneous[2]).T
    points_2d = points_2d[:, :2].reshape((height, width, 2))

    return points_2d


def synthesize_image(reprojected_points, original_image):
    """Synthesize the reprojected image."""
    reprojected_points = np.round(reprojected_points).astype(int)

    height, width = original_image.shape[:2]

    mask = (reprojected_points[:, :, 0] >= 0) & (reprojected_points[:, :, 0] < width) & \
           (reprojected_points[:, :, 1] >= 0) & (reprojected_points[:, :, 1] < height)

    reprojected_image = np.zeros_like(original_image)

    reprojected_image[mask, 0] = original_image[reprojected_points[mask, 1], reprojected_points[mask, 0], 0]
    reprojected_image[mask, 1] = original_image[reprojected_points[mask, 1], reprojected_points[mask, 0], 1]
    reprojected_image[mask, 2] = original_image[reprojected_points[mask, 1], reprojected_points[mask, 0], 2]

    return reprojected_image


def reproject_to_3d(image, depth_map, intrinsic_matrix):
    # First invert the intrinsic matrix
    inverted_intrinsics = np.linalg.inv(intrinsic_matrix)

    # Create an empty array to store the 3D points
    points_3d = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)

    # For each pixel in the image...
    for v in range(image.shape[0]):
        for u in range(image.shape[1]):
            # Construct the 2D homogeneous coordinate
            m = np.array([u, v, 1])

            # Get the depth for this pixel
            Z = depth_map[v, u]

            # Compute the 3D point in the camera coordinate system
            P = np.dot(inverted_intrinsics, m) * Z

            # Store the 3D point
            points_3d[v, u, :] = P

    return points_3d


def reproject_to_2d(points_3d, intrinsic_matrix, original_image):
    # Initialize an empty array to store the reprojected 2D image
    reprojected_image = np.zeros_like(original_image)

    # Get the height and width of the image
    height, width, _ = original_image.shape

    # Iterate over the 3D points
    for y in range(height):
        for x in range(width):
            # Get the 3D point
            P = points_3d[y, x, :]

            # Skip if the point is at infinity or depth is zero (not valid)
            if P[2] == 0:
                continue

            # Project the point back to 2D
            m = np.dot(intrinsic_matrix, P)

            # Homogeneous to Cartesian coordinates
            u, v = round(m[0] / m[2]), round(m[1] / m[2])

            # Check if the reprojected point falls within the image boundaries
            if 0 <= u < width and 0 <= v < height:
                # Copy the pixel value from the original image
                reprojected_image[v, u, :] = original_image[y, x, :]

    return reprojected_image


def simulate_camera_positions(image, depth_map, intrinsic_matrix, baseline=10, num_positions=11, set_num=0):
    # Convert baseline from cm to meters
    baseline /= 100.0

    # Create an array of camera positions along the baseline
    camera_positions = np.linspace(0, baseline, num_positions)

    # Define target directory
    target_dir = f"results/set_{set_num}"

    # For each camera position...
    for i, t in enumerate(camera_positions):
        # Create a translation vector for this camera position
        translation_vector = np.array([t, 0, 0])

        # Use steps 3,4  functions to reproject the image to 3D
        points_3d = reproject_to_3d(image, depth_map, intrinsic_matrix)

        # Apply translation along the x-axis to the 3D points
        points_3d[:, :, 0] -= translation_vector[0]

        # Reproject the translated 3D points back to 2D
        reproject_image = reproject_to_2d(points_3d, intrinsic_matrix, image)

        # Save the reprojected image to the location specified in the ex2 document
        cv2.imwrite(os.path.join(target_dir, "synth_" + str(i + 1) + ".jpg"), reproject_image)


if __name__ == '__main__':

    data = load_data()
    height_census_window = 13
    width_census_window = 25

    for i, (img_left, img_right, intrinsic_matrix, max_disparity) in enumerate(data, 1):
        start_time = t.time()
        print(f"Working on set {i}:")
        print('\tComputing disparity maps...', end='')
        left_disparity_map, right_disparity_map = stereo_algorithm(img_left, img_right, max_disparity,
                                                                   height_census_window,
                                                                   width_census_window)

        print('\tComputing depth maps...', end='')
        left_depth_map = calculate_depth(left_disparity_map, 0.1, intrinsic_matrix[0][0])
        right_depth_map = calculate_depth(right_disparity_map, 0.1, intrinsic_matrix[0][0])

        save_results(i, left_disparity_map, right_disparity_map, left_depth_map, right_depth_map)

        print('\tComputing synthesize images...', end='')
        simulate_camera_positions(img_left, left_depth_map, intrinsic_matrix, set_num=i)

        end_time = t.time()
        print('\t(done in {:.2f}s)'.format(end_time - start_time))

    print("Done")
