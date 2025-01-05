from mpi4py import MPI as mpi
from dataset_manipulation.mnist import matricize_mnist,build_dense_spd_local, build_dense_spd
import numpy as np
from time import perf_counter
from sys import argv, stdout
from matplotlib import pyplot as plt
from scipy.linalg import hadamard
import random
import math
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

def is_square(num):
    if num < 0:
        return False  
    root = math.isqrt(num) 
    return root * root == num

def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

def build_srht_sketching_local(shape: tuple[int, int], num_blocks, block_idx, r_state=None, d_state = None):
    m_local = int(shape[0] / num_blocks)
    l = shape[1]
    if r_state != None:random.setstate(r_state)
    if d_state != None: np.random.set_state(d_state)
    if m_local > 0 and (m_local & (m_local - 1)) != 0:
        raise ValueError
    R = np.zeros((l, m_local))
    for idx, value in enumerate(sorted(random.sample(range(0, m_local), l))):
        R[idx, value] = 1
    H = hadamard(m_local)
    RH = R @ H
    del R, H
    for idx in range(0,block_idx+1):
        Dr = np.random.uniform(-1, 1, (m_local))
        Dl = np.random.uniform(-1, 1, (l))
    sigma_local= np.diag(Dl) @ RH @ np.diag(Dr)
    return np.transpose(sigma_local * (1 / np.sqrt(l)))

def build_gaussian_sketching_local(shape: tuple[int, int], num_blocks, block_idx, d_state = None):
    m_local = int(shape[0] / num_blocks)
    l = shape[1]
    if d_state != None: np.random.set_state(d_state)
    if m_local > 0 and (m_local & (m_local - 1)) != 0:
        raise ValueError
    for idx in range(0,block_idx+1):
        sigma = np.random.normal(0,1,(m_local,l))
    return  sigma * (1 / np.sqrt(l))



if __name__ == '__main__':    

    comm = mpi.COMM_WORLD   
    if not is_square(comm.size) or not is_power_of_two(comm.size): raise ValueError

    if len(argv) > 1: sub_len = int(argv[1])
    else: sub_len = 1024

    if len(argv) > 2: trunc = int(argv[2])
    else:trunc = 100

    if len(argv) > 3: decay = int(argv[3])
    else: decay = 100

    l = trunc + 5
    n = sub_len
    grid_side  = int(np.sqrt(comm.size))
    n_local = n//grid_side
    grid_coord = (comm.rank // grid_side,comm.rank % grid_side)

    mnist_matrix = None
    r_s = None
    d_s = None

    if comm.rank == 0:
        mnist_path = "./datasets/full/mnist.scale.full"
        start = perf_counter()
        mnist_matrix = matricize_mnist(mnist_path)
        mnist_matrix = mnist_matrix[:n]
        print(f"Matricize {perf_counter()-start}")
        r_s = random.getstate()
        d_s = np.random.get_state()

    mnist_matrix = comm.bcast(mnist_matrix,root=0)
    r_s = comm.bcast(r_s,root=0)
    d_s = comm.bcast(d_s,root=0)
    	
    row_comm = comm.Split(color=grid_coord[0], key=comm.rank)
    col_comm = comm.Split(color=grid_coord[1], key=comm.rank)
    diag_comm = comm.Split(color=(grid_coord[0] == grid_coord[1]), key=comm.rank)
    if comm.rank == 0: start = perf_counter()
    matrix_local = build_dense_spd_local(mnist_matrix,decay,grid_coord[0]*n_local, grid_coord[1]*n_local,n_local)
    if comm.rank == 0: 
        print(f"Build local {perf_counter()-start}")
        stdout.flush()
    local_trace = 0
    total_trace = 0
    if (grid_coord[0] == grid_coord[1]): local_trace = np.trace(matrix_local)
    total_trace = diag_comm.reduce(local_trace)
    start = perf_counter()
    sigma_i = build_srht_sketching_local((n,l),grid_side,grid_coord[0],r_state=r_s,d_state=d_s)
    sigma_j = build_srht_sketching_local((n,l),grid_side,grid_coord[1],r_state=r_s,d_state=d_s)
    # sigma_i = build_gaussian_sketching_local((n,l),grid_side,grid_coord[0],d_state=d_s)
    # sigma_j = build_gaussian_sketching_local((n,l),grid_side,grid_coord[1],d_state=d_s)
    matrix_local = matrix_local@sigma_j
    C_local = row_comm.reduce(matrix_local,root=0)
    if grid_coord[1] == 0:
        C_local = C_local.flatten()
    else:
        C_local = None
    C_gather = col_comm.gather(C_local, root=0)

    del C_local

    matrix_local = np.transpose(sigma_i)@matrix_local
    B = comm.reduce(matrix_local,root=0)

    del matrix_local
    os.environ["OPENBLAS_NUM_THREADS"] = "72"
    os.environ["MKL_NUM_THREADS"] = "72"
    os.environ["NUMEXPR_NUM_THREADS"] = "72"
    os.environ["OMP_NUM_THREADS"] = "72"

    if comm.rank == 0:

        C = np.zeros((n_local * grid_side, l))
        for row_idx, flat_matrix in enumerate(C_gather):
                C[row_idx * n_local:(row_idx + 1) * n_local, :] = flat_matrix.reshape((n_local, l))
        del C_gather
        U,S,_ = np.linalg.svd(B)
        S=np.sqrt(S)
        S = U@np.diag(S)
        Z = np.linalg.solve(S,C.transpose()).transpose()
        Q,R = np.linalg.qr(Z)
        U,S,_ = np.linalg.svd(R)
        U = U[:,:trunc]
        S = S[:trunc]
        U = Q@U
        S =np.power(S,2)
        stop = perf_counter()
        norm_tot = (total_trace- np.sum(S))/total_trace
        print(f"{comm.size} {n} {trunc} {decay} {norm_tot} {stop-start}")
        
        


