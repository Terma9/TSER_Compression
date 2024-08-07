import os, socket

if socket.gethostname() != "sim-IdeaPad-5-14ALC05":
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[var] = "40" 

import pandas as pd
import numpy as np

from utils.data_loader import load_from_tsfile_to_dataframe
from utils.regressor_tools import process_data
#import mlflow
#from tsfeatures import tsfeatures
np.set_printoptions(threshold=np.inf)
#pd.set_option('display.max_rows', None)  
#pd.set_option('display.max_columns', None) 
from utils.personal_utils import *
from compression import *
from utils.compression_algos import *

import os
import matplotlib.pyplot as plt


from tsfresh import extract_features, select_features, extract_relevant_features
from tsfresh.feature_extraction import extract_features, EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute


import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)



paths = {
    'AppliancesEnergy':   '/home/simon/TSER/Time_Series_Data/AppliancesEnergy_TRAIN.ts',
    'BeijingPM25Quality': '/home/simon/TSER/Time_Series_Data/BeijingPM25Quality_TRAIN.ts',
    'IEEEPPG':            '/home/simon/TSER/Time_Series_Data/IEEEPPG_TRAIN.ts',
    'NewsTitleSentiment': '/home/simon/TSER/Time_Series_Data/NewsTitleSentiment_TRAIN.ts',
    
    'BenzeneConcentration':       '/home/simon/TSER/Time_Series_Data/BenzeneConcentration_TRAIN.ts',
    'Covid3Month':        '/home/simon/TSER/Time_Series_Data/Covid3Month_TRAIN.ts',
    'FloodModeling1':             '/home/simon/TSER/Time_Series_Data/FloodModeling1_TRAIN.ts',
    'HouseholdPowerConsumption1': '/home/simon/TSER/Time_Series_Data/HouseholdPowerConsumption1_TRAIN.ts'
}

for dataset_id, data_path in paths.items():
    dataset_array, data_y = load_dataset(data_path)
    dataset_id = os.path.basename(data_path).split('_')[0]



    num_dp, len_ts, num_dim = dataset_array.shape

    # Create the ts-values per dimension and convert to df
    array_flatdim = dataset_array.reshape(-1, num_dim).copy()
    column_names_dim = [f"dim_{i+1}" for i in range(num_dim)]
    dataset_df = pd.DataFrame(array_flatdim,columns=column_names_dim)

    # Create the timesteps flattened
    timesteps_flattened = np.tile(np.arange(len_ts), num_dp)


    #Create the id for each datapoint/sample
    ts_ids = [i for i in range(num_dp) for _ in range(len_ts)]

    dataset_df.insert(0, 'timesteps', timesteps_flattened)
    dataset_df.insert(0, 'ts_ids', ts_ids)


    y_ids = [i+1 for i in range(num_dp)]

    y_ser = pd.Series(data_y,index=y_ids)


    extracted = extract_features(timeseries_container=dataset_df,column_id='ts_ids',column_sort="timesteps", default_fc_parameters=EfficientFCParameters())
    impute(extracted)

    print(dataset_id)

    print(f'SHAPE EXTRACTED {extracted.shape}')

    sf = select_features(extracted, y_ser, ml_task='regression')

    print(f'SHAPE SELECTED {sf.shape}')