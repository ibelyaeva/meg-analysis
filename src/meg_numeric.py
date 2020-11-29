import numpy as np
from scipy import sparse


def apply_scaling(data, scalings, verbose=True):
    print (data.dtype)
    data = data * scalings  # F - order
    
    return data
        
        