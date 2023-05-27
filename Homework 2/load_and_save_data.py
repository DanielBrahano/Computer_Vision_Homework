import os
import cv2
import numpy as np


def load_data():
    data = []
    for i in range(1, 6):
        set_folder = f"data/set_{i}"

        # Read the left and right images
        img_left = cv2.imread(os.path.join(set_folder, "im_left.jpg"))
        img_right = cv2.imread(os.path.join(set_folder, "im_right.jpg"))

        # Read the intrinsic matrix from K.txt
        intrinsic_matrix = np.loadtxt(os.path.join(set_folder, "K.txt"))

        # Read the max_disparity from max_disp.txt
        max_disparity = int(np.loadtxt(os.path.join(set_folder, "max_disp.txt")))

        # Append the data for this set to the main list
        data.append((img_left, img_right, intrinsic_matrix, max_disparity))

    return data


def save_results(set_number, left_disparity_map, right_disparity_map, left_depth_map, right_depth_map):
    # Define source and target directories
    source_dir = f"data/set_{set_number}"
    target_dir = f"results/set_{set_number}"

    # Make sure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Copy original images
    for img_name in ["im_left.jpg", "im_right.jpg"]:
        img = cv2.imread(os.path.join(source_dir, img_name))
        cv2.imwrite(os.path.join(target_dir, img_name), img)

    # Save disparity maps
    left_disparity_map_normalized = normalize_image(left_disparity_map)
    right_disparity_map_normalized = normalize_image(right_disparity_map)

    cv2.imwrite(os.path.join(target_dir, "disp_left.jpg"), left_disparity_map_normalized)
    cv2.imwrite(os.path.join(target_dir, "disp_right.jpg"), right_disparity_map_normalized)

    # Save depth maps
    left_depth_map_normalized = normalize_image(left_depth_map)
    right_depth_map_normalized = normalize_image(right_depth_map)

    cv2.imwrite(os.path.join(target_dir, "depth_left.jpg"), left_depth_map_normalized)
    cv2.imwrite(os.path.join(target_dir, "depth_right.jpg"), right_depth_map_normalized)



def normalize_image(image):
    max_value = np.max(image)

    # Normalize the depth map
    normalized_depth_map = (image / max_value) * 255
    return cv2.convertScaleAbs(normalized_depth_map)

