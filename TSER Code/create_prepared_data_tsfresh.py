
import pandas as pd
import numpy as np
from tsfeatures import tsfeatures
import threading
import time

from utils.data_loader import load_from_tsfile_to_dataframe
from utils.regressor_tools import process_data

from compression import compress_dataset

import os # for extracting dataset_id 


from tsfresh import extract_features, select_features, extract_relevant_features
from tsfresh.feature_extraction import extract_features, EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute

import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)




# Small helper Function for loading data and applying compression.
# Return the compressed Dataset. Either as flat_dim if noDecompres, or as (num_dp, len_ts, num_dim) if andDecompress=True

def load_and_compress(data_path: str, compression_type: str, compression_param: float, andDecompress: bool):

    # everyting except last 2 lines is the same as in load_dataset
    data_x, data_y = load_from_tsfile_to_dataframe(data_path, replace_missing_vals_with='NaN')

    min_len = np.inf
    for i in range(len(data_x)):
        x = data_x.iloc[i, :]
        all_len = [len(y) for y in x]
        min_len = min(min(all_len), min_len)

    data_x_p = process_data(data_x, normalise='standard', min_len=min_len)

    dataset_id = os.path.basename(data_path).split('_')[0]

    
    return compress_dataset(data_x_p, dataset_id, andDecompress= andDecompress, compression_type=compression_type, compression_param=compression_param)
    






# Function that returns ready to use df of ts_and_features and features
# adds y as "target" into df 
# If compression_type is None, no compression is applied. Possible compression types: 'dwt', 'dct', 'dft'


# For load and prepare without compression, compression_type = None! -> keep in mind, compression_param = 0 is different than no compression!

def load_and_prepare_everything(data_path: str, compression_type: str, compression_param: float):
    
    data_x, data_y = load_from_tsfile_to_dataframe(data_path, replace_missing_vals_with='NaN')


    # Return length of shortest ts. In our datasets all ts have same length! -> Keep for safety! They say some dimensions have unequal length!
    min_len = np.inf
    for i in range(len(data_x)):
        x = data_x.iloc[i, :]
        all_len = [len(y) for y in x]
        min_len = min(min(all_len), min_len)


    # Returns cuboid with each layer one time series, in each column is a ts of the belonging dimension. The exact timestemps are not given(since not really important!)
    # normalise = 'standard','minmax',None
    data_x_p = process_data(data_x, normalise='standard', min_len=min_len)


    # Extract the dataset_id from the path -> ID is the name of the dataset without _TRAIN.ts or _TEST.ts.
    dataset_id = os.path.basename(data_path).split('_')[0]


    #Apply compression and decompression for approximated values.
    if compression_type != None:
        data_x_p = compress_dataset(data_x_p, dataset_id, andDecompress=True, compression_type=compression_type, compression_param=compression_param)

    
    # Swap the dimensions so that columns are stacked after each other. Copy since swapaxes only returns a view
    #(95, 144, 24) -> (95, 24, 144), first column gets first row etc. One Row is the ts of the belonging dimension.
    data_swapped = data_x_p.swapaxes(1, 2).copy()

    # Reshape to flattened ts. Stack the rows behind the other for each slice.
    data_x_flattend = data_swapped.reshape(data_swapped.shape[0], -1)


    prep_data = pd.DataFrame(data_x_flattend)
    #prep_data['target'] = data_y
    prep_data.columns = prep_data.columns.astype(str) #fwiz or flaml needs string as columns!


    # Up to this point we finished with the flattened ts. Now we calculate the features.

    num_dp, len_ts, num_dim = data_x_p.shape

    # Create the ts-values per dimension and convert to df
    array_flatdim = data_x_p.reshape(-1, num_dim).copy()
    column_names_dim = [f"dim_{i+1}" for i in range(num_dim)]
    dataset_df = pd.DataFrame(array_flatdim,columns=column_names_dim)

    # Create the timesteps flattened
    timesteps_flattened = np.tile(np.arange(len_ts), num_dp)

    #Create the id for each datapoint/sample
    ts_ids = [i for i in range(num_dp) for _ in range(len_ts)]

    dataset_df.insert(0, 'timesteps', timesteps_flattened)
    dataset_df.insert(0, 'ts_ids', ts_ids)

    #y_ids = [i+1 for i in range(num_dp)]
    #y_ser = pd.Series(data_y,index=y_ids)

    
    extracted = extract_features(timeseries_container=dataset_df,column_id='ts_ids',column_sort="timesteps", default_fc_parameters=EfficientFCParameters())
    #selected = select_features(extracted, y_ser, ml_task='regression')
    impute(extracted)


    extracted['target'] = data_y

    # Remove JSON Strings
    extracted.columns = [col.replace('"', '').replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace(',', '') for col in extracted.columns]



    ts_and_features = pd.concat([prep_data, extracted], axis=1)





    return ts_and_features, extracted


# Changed to load only test of houshold and ieeeppg

def main():
    # Don't overwrite stuff! Comment what is not needed!

    # For first Pipeline run


    # TO-DO - Add new Datasets, add naming, save in same folder, move old folder somewhere else!


    data_names = [

        #'AppliancesEnergy',
        #'FloodModeling1',
        #'HouseholdPowerConsumption1',
        'BeijingPM25Quality',

        'IEEEPPG',  
        'Covid3Month',
        'BenzeneConcentration',
        'NewsTitleSentiment',
        

    ]

    source_path = '/home/simon/TSER/Time_Series_Data/'
    dest_path = '/home/simon/TSER/preparedData/'




    for name in data_names:

        # simplify by using .rename
        
        
        data_train_path = source_path + name + "_TRAIN.ts"
        data_test_path = source_path + name + "_TEST.ts"

        
        start_time = time.strftime("%H:%M:%S %p", time.localtime())
        train_data, train_features = load_and_prepare_everything(data_train_path, None, -1)
        end_time = time.strftime("%H:%M:%S %p", time.localtime())
        print(f'Successfull loading of {name} TRAIN. Starttime: {start_time}. Endtime: {end_time}')


        start_time = time.strftime("%H:%M:%S %p", time.localtime())
        test_data, test_features = load_and_prepare_everything(data_test_path, None, -1)
        end_time = time.strftime("%H:%M:%S %p", time.localtime())
        print(f'Successfull loading of {name} TEST. Starttime: {start_time}. Endtime: {end_time}')



        train_data.to_csv(dest_path + name + "_TRAIN" + "_None_" +'_ts_and_features.csv', index=False)
        train_features.to_csv(dest_path + name + "_TRAIN" + "_None_" + '_features.csv', index=False)

        test_data.to_csv(dest_path + name + "_TEST" + "_None_" + '_ts_and_features.csv', index=False)
        test_features.to_csv(dest_path + name + "_TEST" +"_None_" + '_features.csv', index=False)

        
        # Load extra 5 dct for testing in beginning!

        continue

        for tq in ['dct','dft','dwt']:
            
            if name == 'HouseholdPowerConsumption1' and tq == 'dct':
                continue

            

        
            for i in [0.5,0.75,0.85,0.9,0.95,0.97,0.99]:

                start_time = time.strftime("%H:%M:%S %p", time.localtime())
                train_data, train_features = load_and_prepare_everything(data_train_path, tq, i)
                end_time = time.strftime("%H:%M:%S %p", time.localtime())
                print(f'Successfull loading of {name} {i} {tq} TRAIN. Starttime: {start_time}. Endtime: {end_time}')


                start_time = time.strftime("%H:%M:%S %p", time.localtime())
                test_data, test_features = load_and_prepare_everything(data_test_path, tq, i)
                end_time = time.strftime("%H:%M:%S %p", time.localtime())
                print(f'Successfull loading of {name} {i} {tq} TEST. Starttime: {start_time}. Endtime: {end_time}')




                train_data.to_csv(dest_path + name + "_TRAIN" + f"_{tq}_" + str(i) + '_ts_and_features.csv', index=False)
                train_features.to_csv(dest_path + name + "_TRAIN" + f"_{tq}_" + str(i) + '_features.csv', index=False)

                test_data.to_csv(dest_path + name +"_TEST" + f"_{tq}_" + str(i) + '_ts_and_features.csv', index=False)
                test_features.to_csv(dest_path + name +"_TEST" + f"_{tq}_" + str(i) + '_features.csv', index=False)




def print_progress():
    while True:
        current_time = time.localtime()
        formatted_time = time.strftime("%H:%M:%S %p", current_time)
        print("Program is still running. Current Time:", formatted_time)
        time.sleep(300)


if __name__ == "__main__":
    print("Starting program...")
    # Create a thread for printing progress
    progress_thread = threading.Thread(target=print_progress)
    progress_thread.daemon = True  # Daemonize the thread so it exits when the main program finishes
    progress_thread.start()

    main()
    print("Program finished.")

