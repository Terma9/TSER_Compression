import pandas as pd
import numpy as np
from utils.data_loader import load_from_tsfile_to_dataframe
from utils.regressor_tools import process_data
import os
from compression import *


# Load Dataset into Array!
def load_dataset(data_path, norm = "standard"):
    data_x, data_y = load_from_tsfile_to_dataframe(data_path, replace_missing_vals_with='NaN')

    min_len = np.inf
    for i in range(len(data_x)):
        x = data_x.iloc[i, :]
        all_len = [len(y) for y in x]
        min_len = min(min(all_len), min_len)

    data_x_p = process_data(data_x, normalise='standard', min_len=min_len)
    return(data_x_p)








def compute_rmse(array1, array2):
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


# Adds all average rmse of all datapoints in the dataset, then returns average of that
def compute_avg_rmse_of_dataset(dataset_array, dataset_array_comp):
    # calculate rmse per dp, for all datapoints, add and then divide by number of datapoints
    rmse_all_dp = 0
    for i in range(dataset_array.shape[0]):
        rmse_all_dp += compute_rmse_of_datapoint(dataset_array[i,:,:], dataset_array_comp[i,:,:])[0]
    
    return rmse_all_dp/dataset_array.shape[0]



def compute_mae_of_datapoint(matrix1, matrix2):
    mae_all = 0
    mae_array = np.zeros(matrix1.shape[1])
    
    for i in range(matrix1.shape[1]):
        mae = np.mean(np.abs(matrix1[:,i] - matrix2[:,i]))
        mae_array[i] = mae
        mae_all += mae

    return mae_all/matrix1.shape[1], mae_array


def compute_avg_mae_of_dataset(dataset_array, dataset_array_comp):
    mae_all_dp = 0
    for i in range(dataset_array.shape[0]):
        mae_all_dp += compute_mae_of_datapoint(dataset_array[i,:,:], dataset_array_comp[i,:,:])[0]
    
    return mae_all_dp/dataset_array.shape[0]





def test_compression():

    paths = {
    'FloodModeling1': '/home/sim/Desktop/TS Extrinsic Regression/data/FloodModeling1_TEST.ts',
    'AppliancesEnergy':   '/home/sim/Desktop/TS Extrinsic Regression/data/AppliancesEnergy_TEST.ts',
    'BeijingPM25Quality': '/home/sim/Desktop/TS Extrinsic Regression/data/BeijingPM25Quality_TEST.ts',
    'NewsTitleSentiment': '/home/sim/Desktop/TS Extrinsic Regression/data/NewsTitleSentiment_TEST.ts'
    }
    comp_techniques = ['dct','dft','dwt']
    



    for path in paths:
        data_path = paths[path]

        dataset_array = load_dataset(data_path)
        dataset_id = os.path.basename(data_path).split('_')[0]

        print(f"+++ {dataset_id} +++")

        for comp_tq in comp_techniques:
            print(f"$$$ {comp_tq} $$$")

            print("RMSE")
            for i in np.arange(0, 1.04, 0.04):
                decompressed_dataset = compress_dataset(dataset_array.copy(), dataset_id, True, comp_tq, i) 
                print(f"{i:.2f}  {compute_avg_rmse_of_dataset(dataset_array, decompressed_dataset)}")

            print("\n")

            print("Comp-Ratio")
            for i in np.arange(0, 1.04, 0.04):
                decompressed_dataset = compress_dataset(dataset_array.copy(), dataset_id, False, comp_tq, i) 
                print(f"{i:.2f}  {calculateCompRatio(dataset_array, decompressed_dataset)}")

            print("\n")
            print("\n")



























# Are useless, because can directly load the whole dataset and then slice it

def load_single_ts(data_path, datapoint = 0, dim = 0, norm = "standard"):
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