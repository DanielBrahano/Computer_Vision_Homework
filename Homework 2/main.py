import sys
import time as t

import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
---------------------------------COST VOLUME CALCULATION---------------------------------------
'''


def hamming_distance(ch1, ch2):
    return bin(ch1 ^ ch2).count('1')


def calculate_volume_cost(
        census_image_left: np.ndarray,
        census_image_right: np.ndarray,
        max_disparity: int,
        window_size: int = 3
) -> (np.ndarray, np.ndarray):
    # Padding the images
    padding = max_disparity + window_size // 2
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


def apply_census_transform(input_image_l: np.ndarray, input_image_r: np.ndarray, kernel_size: int) -> (
        np.ndarray, np.ndarray):
    """ Census feature extraction. """

    height, width = input_image_l.shape

    # convert to float and normalize
    input_image_l = normalize_data(input_image_l.astype(float))
    input_image_r = normalize_data(input_image_r.astype(float))

    left_census_values = np.zeros(shape=(height, width), dtype=np.uint64)
    right_census_values = np.zeros(shape=(height, width), dtype=np.uint64)

    print('\tComputing left and right census...', end='')
    start_time = t.time()

    # Exclude pixels on the border (they will have no census values)
    for y in range(kernel_size, height - kernel_size):
        for x in range(kernel_size, width - kernel_size):
            # extract block region from left image and compute its census value
            block_l = np.subtract(
                input_image_l[y - kernel_size:y + kernel_size + 1, x - kernel_size:x + kernel_size + 1],
                input_image_l[y, x], dtype=np.float64)
            left_census_values[y, x] = calculate_census(block_l)

            # extract block region from right image and compute its census value
            block_r = np.subtract(
                input_image_r[y - kernel_size:y + kernel_size + 1, x - kernel_size:x + kernel_size + 1],
                input_image_r[y, x], dtype=np.float64)
            right_census_values[y, x] = calculate_census(block_r)

    end_time = t.time()
    print('\t(done in {:.2f}s)'.format(end_time - start_time))

    return left_census_values, right_census_values


def stereo_algorithm(left_image, right_image, max_disparity, window_size=3):
    path1 = 'data/example/im_left.jpg'
    path2 = 'data/example/im_right.jpg'

    # Load images
    im_left = cv2.imread(path1)
    im_right = cv2.imread(path2)

    # Convert the color images to grayscale
    gray_left = cv2.cvtColor(im_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(im_right, cv2.COLOR_BGR2GRAY)

    gray_left, gray_right = apply_census_transform(gray_left, gray_right, window_size)

    cost_left, cost_right = calculate_volume_cost(gray_left, gray_right, max_disparity, window_size)

    disp_map_left = np.argmin(cost_left, axis=2)
    disp_map_right = np.argmin(cost_right, axis=2)

    return disp_map_left, disp_map_right
if __name__ == '__main__':

    from os.path import splitext
    import matplotlib.pyplot as plt

    print('\nLoad images...')
    #img_l = load_img_file('../../examples/data/cones/im_left.jpg')
    #img_r = load_img_file('../../examples/data/cones/im_right.jpg')

    img_l = cv2.imread('data/example/im_left.jpg')
    img_r = cv2.imread('data/example/im_right.jpg')
    dawn = t.time()

    l_disparity_map, r_disparity_map  = stereo_algorithm(img_l, img_r, 77, 3)


    dusk = t.time()
    print('\nTotal execution time = {:.2f}s'.format(dusk - dawn))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Create a figure with 1 row and 2 columns of subplots

    # Show img1 in the first subplot
    axs[0].imshow(l_disparity_map, cmap='gray')  # Use cmap='gray' for grayscale images
    axs[0].axis('off')  # Hide the axes on this subplot
    axs[0].set_title('Image 1')  # Set title for first image

    # Show img2 in the second subplot
    axs[1].imshow(r_disparity_map, cmap='gray')  # Use cmap='gray' for grayscale images
    axs[1].axis('off')  # Hide the axes on this subplot
    axs[1].set_title('Image 2')  # Set title for second image

    plt.show()  # Display the figure with the images