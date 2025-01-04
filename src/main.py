from dataset_manipulation.mnist import matricize_mnist, build_dense_spd
import numpy as np
from time import perf_counter
from sys import argv
from matplotlib import pyplot as plt
from scipy.linalg import hadamard
import random
def build_gaussian_sketching(shape: tuple[int,int]):
    return  np.random.normal(0,1,shape)*(1/np.sqrt(shape[1]))

def build_srht_sketching(shape: tuple[int,int]):
    H = hadamard(shape[0])
    Dr = np.random.uniform(-1,1,(shape[0]))
    Dl = np.random.uniform(-1,1,(shape[1]))
    R = np.zeros((shape[1],shape[0]))
    for idx, value in enumerate(sorted(random.sample(range(0,shape[0]),shape[1]))):
        R[idx, value]=1        
    matrix = np.diag(Dl)@R@H@np.diag(Dr)
    return  np.transpose(matrix*(1/np.sqrt(shape[1])))

def random_truncated_nystrom(matrix: np.ndarray, target_rank:int, oversampling: int, sketching: np.ndarray = np.empty((0))):
    if sketching.shape[0]==0: sketching = build_gaussian_sketching((mnist_matrix.shape[1],target_rank+oversampling))
    C = matrix@sketching
    U,S,_ = np.linalg.svd(np.transpose(sketching)@C)
    S=np.sqrt(S)
    S = U@np.diag(S)
    Z = np.linalg.solve(S,C.transpose()).transpose()
    Q,R = np.linalg.qr(Z)
    U,S,_ = np.linalg.svd(R)
    U = U[:,:target_rank]
    S = S[:target_rank]
    return Q@U,np.power(S,2)

def random_truncated_svd(matrix: np.ndarray, target_rank:int, oversampling: int, sketching: np.ndarray = np.empty((0))):
    if sketching.shape[0]==0: sketching = build_gaussian_sketching((mnist_matrix.shape[1],target_rank+oversampling))
    Q, _ = np.linalg.qr( matrix @ sketching)
    P, R = np.linalg.qr(np.transpose(np.transpose(Q)@matrix),mode='reduced')
    U,S,Vh = np.linalg.svd(np.transpose(R), full_matrices=False)
    print( U.shape, S.shape, Vh.shape,P.shape,R.shape)
    U  = U[:,:target_rank]
    S  = S[: target_rank]
    Vh = Vh[:target_rank, :]
    print( U.shape, S.shape, Vh.shape,P.shape,R.shape)
    return Q@U,S,Vh@np.transpose(P)

if __name__ =='__main__':
    if len(argv) > 1:
        sub_len = int(argv[1])
    else:
        sub_len = 1000
    if len(argv) > 2:
        trunc = int(argv[2])
    else:
        trunc = 100
    if len(argv) > 3:
        decay = int(argv[3])
    else:
        decay = 100
    print(sub_len)

    mnist_path = "./datasets/full/mnist.scale.full"
    mnist_matrix = matricize_mnist(mnist_path)
    mnist_matrix = build_dense_spd(mnist_matrix[0:sub_len],decay)

    start_random = perf_counter()
    sketch = build_srht_sketching((mnist_matrix.shape[1],trunc + 5))
    U,S = random_truncated_nystrom(mnist_matrix,trunc,5,sketching=sketch)
    stop_random = perf_counter()
    print(np.sum(S))
    print(f"Nystrom t: {stop_random-start_random}\n")

