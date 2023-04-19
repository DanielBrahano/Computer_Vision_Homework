import os

import cv2


def save_relative_image(image, type, puzzle_num, piece_num):

    temp_path = 'my_results/puzzle_' + str(type) + '_' + str(puzzle_num) + '/piece_' + str(piece_num) + '.jpeg'

    output_dir = 'my_results/puzzle_' + str(type) + '_' + str(puzzle_num)

    # Create the directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the file path
    path = os.path.join(output_dir, '/piece_' + str(piece_num) + '.jpeg')
    path = 'my_results/puzzle_' + str(type) + '_' + str(puzzle_num) + '/piece_' + str(piece_num) + '.jpeg'

    cv2.imwrite(path, image)


def save_solution_image(image, type, puzzle_num, assmbled_piece, total_pieces):
    path = 'my_results/puzzle_' + str(type) + '_' + str(puzzle_num) + '/solution_' + str(
        assmbled_piece) + '_' + str(total_pieces) + '.jpeg'
    cv2.imwrite(path, image)
