from matplotlib import pyplot as plt
import numpy as np
import os

def matricize_mnist(mnist_path: str):
    if os.path.isfile(mnist_path):
        with open(mnist_path,"r") as dataset_handle:
            data_sketch = dataset_handle.readlines()
            data = np.zeros((len(data_sketch),784),dtype=np.float64)
            for idx, line in enumerate(data_sketch):
                data[idx] = parse_mnist_line(line)
            return data
    else:
        print(f"File {mnist_path} does not exists.")
        raise ValueError
def parse_mnist_line( line: str):
    data = np.zeros((784),dtype=np.float64);
    data_sketch = line.split(" ")[1:-1]
    for element in data_sketch:
        idx,val = element.split(":")
        data[int(idx)] = float(val)
    return data

def plot_dataline_mnist(row: int, mnist_path: str):  
    if os.path.isfile(mnist_path):
        with open(mnist_path,"r") as dataset_handle:
            data_sketch = dataset_handle.readlines()
            plt.imshow(parse_mnist_line(data_sketch[row]).reshape((28,28)), cmap='gray')  # Grayscale color map
            plt.colorbar()  # Adds a color bar to the side
            plt.show() 
    else:
        print(f"File {mnist_path} does not exists.")

def build_dense_spd(matrix: np.ndarray, decay_factor: np.float64):
    dists = np.sum(matrix**2, axis=1).reshape(-1, 1) + np.sum(matrix**2, axis=1) - 2 * np.dot(matrix, matrix.T)
    output_matrix = np.exp(-dists / (decay_factor**2))
    return output_matrix