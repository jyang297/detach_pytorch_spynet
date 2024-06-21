import torch
import matplotlib.pyplot as plt
import time



def update_block_position(matrix, position, block_size, block_color, matrix_size):
    matrix.fill_(0)
    x, y = position
    x_end = min(x + block_size, matrix_size)
    y_end = min(y + block_size, matrix_size)
    for i in range(3):
        matrix[x:x_end, y:y_end, i] = block_color[i]
    return matrix


def matrixs_moving(num_frames=20, initial_position=(0, 0), matrix_size=100, block_size=10):
    list_matrix = []

    position = list(initial_position)

    block_color = torch.rand(3)
    matrix = torch.zeros((matrix_size, matrix_size, 3))
    matrix = update_block_position(matrix, position, block_size, block_color, matrix_size)
    list_matrix.append(matrix.clone().permute(2,0,1))
    num_steps = num_frames


    for i in range(num_steps):
        position[0] = (position[0] + 5) % matrix_size
        position[1] = (position[1] + 5) % matrix_size
        matrix = update_block_position(matrix, position, block_size, block_color, matrix_size)
        list_matrix.append(matrix.clone().permute(2,0,1))

    return torch.stack(list_matrix)
