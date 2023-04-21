import cv2
import numpy as np
from matplotlib import pyplot as plt
import re
import os

matrix_affine_1_path = 'puzzles/puzzle_affine_1/warp_mat_1__H_521__W_760_.txt'
matrix_affine_2_path = 'puzzles/puzzle_affine_2/warp_mat_1__H_537__W_735_.txt'
matrix_affine_3_path = 'puzzles/puzzle_affine_3/warp_mat_1__H_497__W_741_.txt'
matrix_affine_4_path = 'puzzles/puzzle_affine_4/warp_mat_1__H_457__W_808_.txt'
matrix_affine_5_path = 'puzzles/puzzle_affine_5/warp_mat_1__H_510__W_783_.txt'
matrix_affine_6_path = 'puzzles/puzzle_affine_6/warp_mat_1__H_522__W_732_.txt'
matrix_affine_7_path = 'puzzles/puzzle_affine_7/warp_mat_1__H_511__W_732_.txt'
matrix_affine_8_path = 'puzzles/puzzle_affine_8/warp_mat_1__H_457__W_811_.txt'
matrix_affine_9_path = 'puzzles/puzzle_affine_9/warp_mat_1__H_481__W_771_.txt'
matrix_affine_10_path = 'puzzles/puzzle_affine_10/warp_mat_1__H_507__W_771_.txt'

matrix_homography_1_path = 'puzzles/puzzle_homography_1/warp_mat_1__H_549__W_699_.txt'
matrix_homography_2_path = 'puzzles/puzzle_homography_2/warp_mat_1__H_513__W_722_.txt'
matrix_homography_3_path = 'puzzles/puzzle_homography_3/warp_mat_1__H_502__W_760_.txt'
matrix_homography_4_path = 'puzzles/puzzle_homography_4/warp_mat_1__H_470__W_836_.txt'
matrix_homography_5_path = 'puzzles/puzzle_homography_5/warp_mat_1__H_457__W_811_.txt'
matrix_homography_6_path = 'puzzles/puzzle_homography_6/warp_mat_1__H_464__W_815_.txt'
matrix_homography_7_path = 'puzzles/puzzle_homography_7/warp_mat_1__H_488__W_760_.txt'
matrix_homography_8_path = 'puzzles/puzzle_homography_8/warp_mat_1__H_499__W_760_.txt'
matrix_homography_9_path = 'puzzles/puzzle_homography_9/warp_mat_1__H_490__W_816_.txt'
matrix_homography_10_path = 'puzzles/puzzle_homography_10/warp_mat_1__H_506__W_759_.txt'

all_affine_warp_matrices_paths = [matrix_affine_1_path, matrix_affine_2_path, matrix_affine_3_path,
                                  matrix_affine_4_path, matrix_affine_5_path, matrix_affine_6_path,
                                  matrix_affine_7_path, matrix_affine_8_path, matrix_affine_9_path,
                                  matrix_affine_10_path]

all_homography_warp_matrices_paths = [matrix_homography_1_path, matrix_homography_2_path, matrix_homography_3_path,
                                      matrix_homography_4_path, matrix_homography_5_path, matrix_homography_6_path,
                                      matrix_homography_7_path, matrix_homography_8_path, matrix_homography_9_path,
                                      matrix_homography_10_path]

num_of_pieces_in_puzzle_affine = np.array([2, 5, 8, 11, 14, 25, 38, 36, 57, 62])
num_of_pieces_in_puzzle_homography = np.array([4, 5, 6, 12, 16, 21, 24, 34, 35, 59])


def get_affine_images():
    """
    :return: all_affine_images: array which each entry is array of piece for a puzzle
    """
    all_affine_images = []
    num_of_pieces_in_puzzle_affine = np.array([2, 5, 8, 11, 14, 25, 38, 36, 57, 62])
    for i in range(len(num_of_pieces_in_puzzle_affine)):

        images_in_piece = []
        for j in range(1, num_of_pieces_in_puzzle_affine[i] + 1):
            img = cv2.imread('puzzles/puzzle_affine_' + str(i + 1) + '/pieces/piece_' + str(j) + '.jpg')
            images_in_piece.append(img)
        all_affine_images.append(images_in_piece)

    return all_affine_images


def get_homography_warp_matrix(homography_mat_path):
    """
    Get homography matrix.
    :param homography_mat_path: path to txt file representing matrix
    :return: matrix: homography matrix
    """
    with open(homography_mat_path, 'r') as f:
        lines = f.readlines()

    matrix = []
    for line in lines:
        row = [float(x) for x in line.split()]
        matrix.append(row)

    matrix = np.array(matrix)
    #print(matrix)

    return matrix


def get_affine_warp_matrix(affine_mat_path):
    """
    Get homography matrix.
    :param mat_path: path to txt file representing matrix
    :return: matrix: affine matrix
    """
    # Read the text file
    with open(affine_mat_path, 'r') as f:
        lines = f.readlines()

    # Parse the lines to create a list of lists
    matrix = []
    for line in lines[:-1]:
        row = [float(x) for x in line.split()]
        matrix.append(row)

    # Convert the list of lists to a NumPy array
    matrix = np.array(matrix)

    # Print the resulting matrix
    #print(matrix)
    return matrix


def get_height_width(mat_path):
    """
    Get height and width of final puzzle.
    :param mat_path: path to txt file representing matrix
    :return:
    '"""
    match = re.search(r"H_(\d+).*W_(\d+)", mat_path)

    if match:
        XXX = int(match.group(1))
        YYY = int(match.group(2))
        #print(XXX, YYY)
        return XXX, YYY

    else:
        print(None)
        return None


def get_list_height_width(list_of_mat_paths):
    """
    :param list_of_mat_paths:  list of paths to matrices
    :return: list of heights and widths
    """
    list_height_width = []
    for mat_path in list_of_mat_paths:
        (height, width) = get_height_width(mat_path)
        list_height_width.append((height, width))
    return list_height_width


def get_list_homography_matrices(list_of_mat_paths):
    """
    :param list_of_mat_paths: list of paths to matrices
    :return: list of warp affine matrices
    """
    list_warp_homography_matrices = []
    for mat_path in list_of_mat_paths:
        matrix = get_homography_warp_matrix(mat_path)
        list_warp_homography_matrices.append(matrix)
    return list_warp_homography_matrices


def get_list_affine_matrices(list_of_mat_paths):
    """
    :param list_of_mat_paths: list of paths to matrices
    :return: list of warp affine matrices
    """
    list_warp_affine_matrices = []
    for mat_path in list_of_mat_paths:
        matrix = get_affine_warp_matrix(mat_path)
        list_warp_affine_matrices.append(matrix)
    return list_warp_affine_matrices


def get_affine_images_and_all_info():
    # get list of final warping matrices
    all_affine_warp_mat = get_list_affine_matrices(all_affine_warp_matrices_paths)

    # get list of heights and widths for each puzzle
    heights_widths_list = get_list_height_width(all_affine_warp_matrices_paths)

    all_affine_images = []
    num_of_pieces_in_puzzle_affine = np.array([2, 5, 8, 11, 14, 25, 38, 36, 57, 62])
    for i in range(len(num_of_pieces_in_puzzle_affine)):

        images_in_piece = []
        for j in range(1, num_of_pieces_in_puzzle_affine[i] + 1):
            img = cv2.imread('puzzles/puzzle_affine_' + str(i + 1) + '/pieces/piece_' + str(j) + '.jpg')
            images_in_piece.append(img)
        height, width = heights_widths_list[i]
        final_warp_mat = all_affine_warp_mat[i]
        all_affine_images.append((images_in_piece, height, width, final_warp_mat))

    return all_affine_images


def get_homography_images_and_all_info():
    # get list of final warping matrices
    all_homography_warp_mat = get_list_homography_matrices(all_homography_warp_matrices_paths)

    # get list of heights and widths for each puzzle
    heights_widths_list = get_list_height_width(all_homography_warp_matrices_paths)

    all_homography_images = []
    num_of_pieces_in_puzzle_homography = np.array([4, 5, 6, 12, 16, 21, 24, 34, 35, 59])
    for i in range(len(num_of_pieces_in_puzzle_affine)):

        images_in_piece = []
        for j in range(1, num_of_pieces_in_puzzle_homography[i] + 1):
            img = cv2.imread('puzzles/puzzle_homography_' + str(i + 1) + '/pieces/piece_' + str(j) + '.jpg')
            images_in_piece.append(img)
        height, width = heights_widths_list[i]
        final_warp_mat = all_homography_warp_mat[i]
        all_homography_images.append((images_in_piece, height, width, final_warp_mat))

    return all_homography_images
