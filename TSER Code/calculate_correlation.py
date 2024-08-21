import numpy as np
import matplotlib.pyplot as plt

#from create_prepared_data_tsfresh import *
from compression import calculateCompRatio, compress_dataset
from utils.personal_utils import *

import matplotlib.pyplot as plt






# Calculate the correlation, we get a dict looking like: correlations[('AppliancesEnergy', 'dft', 0.5)] -> correlation
dropout_values = [None, 0.5, 0.75, 0.85, 0.9, 0.95, 0.97, 0.99]
log_path = '/home/sim/Desktop/TS Extrinsic Regression/All-Logs/' 

ds_names = [
    #'NewsTitleSentiment',
    'BenzeneConcentration',
    'BeijingPM25Quality',
    'IEEEPPG',
    'FloodModeling1',
    'HouseholdPowerConsumption1',
    'Covid3Month',
    'AppliancesEnergy',
]


tqs = ['dct', 'dft', 'dwt']

prepared_dfs = {}
correlations = {}

for ds_name in ds_names:
    path_to_runs = log_path + f'{ds_name}_Runs/'



    features_raw = pd.read_parquet('/home/sim/Desktop/TS Extrinsic Regression/features_dfs/' + ds_name + f'/NONE_NONE_{ds_name}_features_TEST')
    #features_raw = load_and_prepare_everything(all_ds[ds_name], None, None)[1].drop('target', axis=1)
    
    path_to_pred = path_to_runs + f'NONE_{ds_name}_20min_Flaml_f/predictions.npy'
    prediction_raw = np.load(path_to_pred)
    features_raw['prediction'] = prediction_raw


    for tq in tqs:
        prepared_dfs[(ds_name,tq,None)] = features_raw.copy() # bc different features for each technique

        for dval in dropout_values[1:]:
            

            features = pd.read_parquet('/home/sim/Desktop/TS Extrinsic Regression/features_dfs/' + ds_name + f'/{dval}_{tq}_{ds_name}_features_TEST')
            #features = load_and_prepare_everything(all_ds[ds_name], tq, dval)[1].drop('target', axis=1)
            
            path_to_pred = path_to_runs + f'{dval}_{tq}_{ds_name}_20min_Flaml_f/predictions.npy'

            with open(path_to_pred, 'r') as file:
                prediction = np.load(path_to_pred)


            features['prediction'] = prediction
            prepared_dfs[(ds_name,tq, dval)] = features.copy()


            # Add Subtraction and Correlation
            sub_df = (features_raw - features)
            corrs = sub_df.corrwith(sub_df['prediction'])
            correlations[(ds_name,tq,dval)] = corrs

            print(f'{ds_name}, {tq}, {dval} correlation calculated')
            





# Get Average Correlation of absolute value over each (ds_name, tq)
# The Entries are Series object, the row name is the featuer name!
corr_ds_name_tq = {}


for ds_name in ds_names:
    

    for tq in tqs:

        # Initalize a df with same columns, rows but zero in each field
        avg_corr = correlations[(ds_name,tq,0.5)].copy()
        avg_corr[:] = 0
        #print(avg_corr)

        for d_val in dropout_values[1:]:
            curr_corr_df = correlations[(ds_name,tq,d_val)].abs()  # take abs value
            avg_corr = curr_corr_df + avg_corr
        
        avg_corr = avg_corr / len(dropout_values[1:])
    
        
        corr_ds_name_tq[(ds_name, tq)] = avg_corr.sort_values(ascending=False)



corr_ds_name = {}
for ds_name in ds_names:
    corr_ds_name[ds_name] = corr_ds_name_tq[(ds_name, 'dct')]
    corr_ds_name[ds_name] += corr_ds_name_tq[(ds_name, 'dft')]
    corr_ds_name[ds_name] += corr_ds_name_tq[(ds_name, 'dwt')]
    
    corr_ds_name[ds_name] = corr_ds_name[ds_name].sort_values(ascending=False) / 3
    


# Get Correlation over each dataset

for ds_name in ds_names:
    print(ds_name )
    print(corr_ds_name[(ds_name)])
    print()




# Print for Each Combination the top 5 features
for ds_name in ds_names:
    for tq in tqs:
        print(ds_name + " " + tq)
        print(corr_ds_name_tq[(ds_name, tq)].head(6))
        print()