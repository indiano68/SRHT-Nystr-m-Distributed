from matplotlib import pyplot as plt
import numpy as np
import os


def matricize_mnist(mnist_path: str):

    if os.path.isfile(mnist_path):
        with open(mnist_path, "r") as dataset_handle:
            data_sketch = dataset_handle.readlines()
            data = np.zeros((len(data_sketch), 784), dtype=np.float64)
            for idx, line in enumerate(data_sketch):
                data[idx] = parse_mnist_line(line)
            return data
    else:
        print(f"File {mnist_path} does not exists.")
        raise ValueError


def parse_mnist_line(line: str):
    data = np.zeros((784), dtype=np.float64)
    data_sketch = line.split(" ")[1:-1]
    for element in data_sketch:
        idx, val = element.split(":")
        data[int(idx)] = float(val)
    return data


def plot_dataline_mnist(row: int, mnist_path: str):
    if os.path.isfile(mnist_path):
        with open(mnist_path, "r") as dataset_handle:
            data_sketch = dataset_handle.readlines()
            plt.imshow(
                parse_mnist_line(data_sketch[row]).reshape((28, 28)), cmap="gray"
            )  # Grayscale color map
            plt.colorbar()  # Adds a color bar to the side
            plt.show()
    else:
        print(f"File {mnist_path} does not exists.")


def build_dense_spd(matrix: np.ndarray, decay_factor: np.float64):
    dists = (
        np.sum(matrix**2, axis=1).reshape(-1, 1)
        + np.sum(matrix**2, axis=1)
        - 2 * np.dot(matrix, matrix.T)
    )
    output_matrix = np.exp(-dists / (decay_factor**2))
    return output_matrix


def __stub_build_dense_spd_local(
    matrix: np.ndarray,
    decay_factor: np.float64,
    i_start: int,
    j_start: int,
    length: int,
):
    matrix_out = np.zeros((length, length))
    for i in range(i_start, i_start + length):
        for j in range(j_start, j_start + length):
            matrix_out[i - i_start, j - j_start] = np.exp(
                -1
                * np.power(np.linalg.norm(matrix[i] - matrix[j]), 2)
                / np.power(decay_factor, 2)
            )
    return matrix_out


def build_dense_spd_local(
    matrix: np.ndarray,
    decay_factor: np.float64,
    i_start: int,
    j_start: int,
    length: int,
):
    i_norm = np.sum(matrix[i_start : i_start + length, :] ** 2, axis=1)[
        :, np.newaxis
    ]  # Shape (n, 1)
    j_norm = np.sum(matrix[j_start : j_start + length, :] ** 2, axis=1)[
        np.newaxis, :
    ]  # Shape (1, m)
    dist_squared = (
        i_norm
        + j_norm
        - 2
        * np.dot(
            matrix[i_start : i_start + length, :],
            matrix[j_start : j_start + length, :].T,
        )
    )  # Shape (n, m)
    matrix_out = np.exp(-dist_squared / (decay_factor**2))
    return matrix_out
