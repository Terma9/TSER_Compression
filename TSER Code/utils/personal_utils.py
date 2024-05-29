# DCT or DFT Compression

#->> both methos shuld be alright, but need to be field tested!
import pandas as pd
import numpy as np
from utils.data_loader import load_from_tsfile_to_dataframe
from utils.regressor_tools import process_data




def load_single_ts(data_path, datapoint = 0, dim = 0, norm = None):
    data_x, data_y = load_from_tsfile_to_dataframe(data_path, replace_missing_vals_with='NaN')

    min_len = np.inf
    for i in range(len(data_x)):
        x = data_x.iloc[i, :]
        all_len = [len(y) for y in x]
        min_len = min(min(all_len), min_len)
    data_x_p = process_data(data_x, normalise= norm, min_len=min_len)
    return(data_x_p[datapoint,:,dim])


# returns dp in matrix form, column up to down ts, colums are the dimensions
def load_datapoint(data_path, datapoint = 0, norm = None):
    data_x, data_y = load_from_tsfile_to_dataframe(data_path, replace_missing_vals_with='NaN')
    min_len = np.inf
    for i in range(len(data_x)):
        x = data_x.iloc[i, :]
        all_len = [len(y) for y in x]
        min_len = min(min(all_len), min_len)
    data_x_p = process_data(data_x, normalise= norm, min_len=min_len)
    return(data_x_p[datapoint,:,:])


def 






def compute_rmse(array1, array2):
    """
    Compute the Root Mean Squared Error (RMSE) between two arrays.

    Returns:
    float: The RMSE value.
    """
    # Ensure the arrays have the same length
    if array1.shape != array2.shape:
        raise ValueError("The input arrays must have the same shape")
    
    return np.sqrt(np.mean((array1 - array2) ** 2))


# Input: Matrices with ts in columns up to down, each column is one dimension
# Returns mean RMSE of all columns, and RMSE of each column



def compute_rmse_of_datapoint(matrix1, matrix2):
    rmse_all = 0
    rmse_array = np.zeros(matrix1.shape[1])
    
    for i in range(matrix1.shape[1]):
        rmse = np.sqrt(np.mean((matrix1[:,i] - matrix2[:,i]) ** 2))
        rmse_array[i] = rmse
        rmse_all += rmse

    return rmse_all/matrix1.shape[1], rmse_array
