
import pandas as pd
import numpy as np
from tsfeatures import tsfeatures
import threading
import time

from utils.data_loader import load_from_tsfile_to_dataframe
from utils.regressor_tools import process_data

from compression import compress_dataset

import os # for extracting dataset_id 




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
    prep_data['target'] = data_y
    prep_data.columns = prep_data.columns.astype(str) #fwiz or flaml needs string as columns!


    # Up to this point we finished with the flattened ts. Now we calculate the features.


    #data_x_p = data_x_p[0:2,...]

    num_datapoints = data_x_p.shape[0]
    len_timeseries = data_x_p.shape[1]
    num_dimensions = data_x_p.shape[2]

    # 38 since tsfeatures returns 38 features
    num_features = 38

    all_features = np.ndarray((num_datapoints, num_features * num_dimensions))

    for i in range(0, num_datapoints):
        start_index = 0

        for j in range(0, num_dimensions):
            curr_ts = data_x_p[i,:,j]

            # Apply tsfeatures
            #print(curr_ts.size)
            timeseries_df = pd.DataFrame({'unique_id' : np.ones(len_timeseries),'ds': np.arange(0,len_timeseries) , 'y': curr_ts})
            

            #!!
            feature_array = np.zeros((1,))
            
            #tsfeatures(timeseries_df, freq=1).fillna(0).values

            #print(feature_array.size)
            #print(np.isnan(feature_array).sum())

            end_index = start_index + feature_array.size
            all_features[i, start_index: end_index] = feature_array
            start_index = end_index
        

    all_features = pd.DataFrame(all_features)

    # name the features, important for fwiz and flaml
    for i, col in enumerate(all_features.columns):
        # Generate the new column name
        new_col_name = 'f' + str(i + 1)
        # Rename the column
        all_features.rename(columns={col: new_col_name}, inplace=True)


    all_features['target'] = data_y


    #!!
    #ts_and_features = pd.concat([prep_data.drop(columns=['target']), all_features], axis=1)


    return all_features, all_features




def main():
    # Don't overwrite stuff! Comment what is not needed!

    # For first Pipeline run

    data_names = [
        'BeijingPM25Quality',
        'FloodModeling1',
        'AppliancesEnergy',
        'Covid3Month'
        
    ]

    source_path = '/home/simon/TSER/Time_Series_Data/'

    for name in data_names:

        # simplify by using .rename
        data_train_path = source_path + name + "_TRAIN.ts"
        data_test_path = source_path + name + "_TEST.ts"

        print("load and prepare "+ name)

        train_data, train_features = load_and_prepare_everything(data_train_path, None, -1)



        test_data, test_features = load_and_prepare_everything(data_test_path, None, -1)

        dest_path = '/home/simon/TSER/preparedData/'



        train_data.to_csv(dest_path + name + "_TRAIN" + "_None_" +'_ts_and_features.csv', index=False)
        train_features.to_csv(dest_path + name + "_TRAIN" + "_None_" + '_features.csv', index=False)

        test_data.to_csv(dest_path + name + "_TEST" + "_None_" + '_ts_and_features.csv', index=False)
        test_features.to_csv(dest_path + name + "_TEST" +"_None_" + '_features.csv', index=False)


        # Load extra 5 dct for testing in beginning!
        if name == 'AppliancesEnergy':

            for i in [0.5,0.75,0.85,0.95,0.99]:

                train_data, train_features = load_and_prepare_everything(data_train_path, 'dct', i)
                test_data, test_features = load_and_prepare_everything(data_test_path, 'dct', i)


                train_data.to_csv(dest_path + name + "_TRAIN" + "_dct_" + str(i) + '_ts_and_features.csv', index=False)
                train_features.to_csv(dest_path + name + "_TRAIN" + "_dct_" + str(i) + '_features.csv', index=False)

                test_data.to_csv(dest_path + name +"_TEST" + "_dct_" + str(i) + '_ts_and_features.csv', index=False)
                test_features.to_csv(dest_path + name +"_TEST" + "_dct_" + str(i) + '_features.csv', index=False)





    print("successfull loading")


def print_progress():
    while True:
        current_time = time.localtime()
        formatted_time = time.strftime("%H:%M:%S %p", current_time)
        print("Program is still running...")
        print("Current Time =", formatted_time)
        time.sleep(300)


if __name__ == "__main__":
    print("Starting program...")
    # Create a thread for printing progress
    progress_thread = threading.Thread(target=print_progress)
    progress_thread.daemon = True  # Daemonize the thread so it exits when the main program finishes
    progress_thread.start()

    main()
    print("Program finished.")

