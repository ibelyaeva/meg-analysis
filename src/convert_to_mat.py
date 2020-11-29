import numpy as np
from scipy.io import savemat
from scipy.io import loadmat

def numpy_to_mat(x, file_path):
    key = {}
    key['data'] = x
    savemat(file_path, key)
    
def mat_numpy(file_path):
    result = loadmat(file_path)
    return result