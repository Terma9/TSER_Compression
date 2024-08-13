
import os, socket

if socket.gethostname() != "sim-IdeaPad-5-14ALC05":
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[var] = "40" 


import argparse
from utils.personal_utils import *
from create_prepared_data_tsfresh import *



ds_names = [
'NewsTitleSentiment',
'HouseholdPowerConsumption2',
'FloodModeling2',
'BeijingPM10Quality'


#'AppliancesEnergy',
#'HouseholdPowerConsumption1',
#'FloodModeling1',
#'Covid3Month',
#'IEEEPPG',
#'BenzeneConcentration',
#'BeijingPM25Quality',



]


#path = 

path_ts = '/home/simon/TSER/Time_Series_Data/'

# Create folder to save the outputs
#current_directory = os.getcwd()
#parent_directory = os.path.dirname(current_directory)


folder_path = '/home/simon/TSER/' + 'features_dfs'


if not os.path.exists(folder_path):
    os.makedirs(folder_path)


for ds_name in ds_names:
    #!! I add each solution and overwrite in case

    ds_path = os.path.join(folder_path, ds_name)
    if not os.path.exists(ds_path):
        os.makedirs(ds_path)


    data_train_path = path_ts + ds_name + "_TRAIN.ts"
    data_test_path = path_ts + ds_name + "_TEST.ts"

    _ , train_features = load_and_prepare_everything(data_train_path, None, -1)
    _ , test_features = load_and_prepare_everything(data_test_path, None, -1)



    train_features.to_parquet(os.path.join(ds_path, f'NONE_NONE_{ds_name}_features_TRAIN'), compression='gzip') #brotli compression even more efficient, but slower!
    test_features.to_parquet(os.path.join(ds_path, f'NONE_NONE_{ds_name}_features_TEST'), compression='gzip')

    for tq in ['dct', 'dft', 'dwt']:
        for i in [0.5, 0.75, 0.85, 0.9, 0.95, 0.97, 0.99]:

            _ , train_features = load_and_prepare_everything(data_train_path, tq, i)
            _ , test_features = load_and_prepare_everything(data_test_path, tq, i)



            train_features.to_parquet(os.path.join(ds_path, f'{i}_{tq}_{ds_name}_features_TRAIN'), compression='gzip')
            test_features.to_parquet(os.path.join(ds_path, f'{i}_{tq}_{ds_name}_features_TEST'), compression='gzip')