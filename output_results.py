import os
import cv2


def save_relative_image(image, type, puzzle_num, piece_num):
    output_dir = 'my_results/puzzle_' + str(type) + '_' + str(puzzle_num)

    # Create the directory if it does not exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    path = 'my_results/puzzle_' + str(type) + '_' + str(puzzle_num) + '/piece_' + str(piece_num) + '_relative' + '.jpeg'

    cv2.imwrite(path, image)


def save_solution_image(image, type, puzzle_num, assmbled_piece, total_pieces):
    path = 'my_results/puzzle_' + str(type) + '_' + str(puzzle_num) + '/solution_' + str(
        assmbled_piece) + '_' + str(total_pieces) + '.jpeg'
    cv2.imwrite(path, image)


def save_coverage_count(image_type, puzzle_num, fig, coverage_count):
    directory = f"my_results/puzzle_{image_type}_{puzzle_num}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    path = os.path.join(directory, 'coverage_count.jpeg')
    fig.savefig(path, format='jpeg', dpi=300)
