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


#import warnings
#warnings.simplefilter("ignore")
#warnings.filterwarnings("ignore")
#warnings.filterwarnings("ignore", category=RuntimeWarning)


path_server = '/home/simon/TSER/preparedData/FloodModeling1_TEST_dct_0.5_features.csv'


paths = {
    'AppliancesEnergy':   '/home/simon/TSER/Time_Series_Data/AppliancesEnergy_TRAIN.ts',
    'BeijingPM25Quality': '/home/simon/TSER/Time_Series_Data/BeijingPM25Quality_TRAIN.ts',
    'IEEEPPG':            '/home/simon/TSER/Time_Series_Data/IEEEPPG_TRAIN.ts',
    'NewsTitleSentiment': '/home/simon/TSER/Time_Series_Data/NewsTitleSentiment_TRAIN.ts',
    'Covid3Month':        '/home/simon/TSER/Time_Series_Data/Covid3Month_TRAIN.ts',

    'BenzeneConcentration':       '/home/simon/TSER/Time_Series_Data/BenzeneConcentration_TRAIN.ts',
    'FloodModeling1':             '/home/simon/TSER/Time_Series_Data/FloodModeling1_TRAIN.ts',
    'HouseholdPowerConsumption1': '/home/simon/TSER/Time_Series_Data/HouseholdPowerConsumption1_TRAIN.ts'
}


from agluon import run_agluon



ds_name = 'AppliancesEnergy'
features_path = '/home/sim/Desktop/TS Extrinsic Regression/features_dfs/' + ds_name + '/'


train_features = pd.read_parquet(features_path + f'NONE_NONE_{ds_name}_features_TRAIN')
test_features = pd.read_parquet(features_path + f'NONE_NONE_{ds_name}_features_TEST')


run_agluon('Test', 'test_run', 100, train_features, test_features)











#for i in np.arange(0, 1.04, 0.04):
#   decompressed_dataset = compress_dataset(dataset_array.copy(), dataset_id, True, "dct", i) 
#    print(compute_avg_rmse_of_dataset(dataset_array, decompressed_dataset))
#dataset_array = np.random.randint(0, 10, size=(3, 10, 3))



#for i in np.arange(0, 1.04, 0.04):
#    decompressed_dataset = compress_dataset(dataset_array.copy(), dataset_id, False, "cpt", i) 
#    print("drop_out value: ",round(i,2),"comp_ratio", calculateCompRatio(dataset_array, decompressed_dataset))
#    print("drop_out value: ",round(i,2),"comp_ratio", calculateCompRatio(dataset_array, decompressed_dataset))

    #print(calculateCompRatio(decompressed_dataset, np.zeros((42,144,24)),True))