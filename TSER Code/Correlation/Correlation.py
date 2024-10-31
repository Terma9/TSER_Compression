import numpy as np
import pandas as pd
import pickle



# Calculate the correlation, we get a dict looking like: correlations[('AppliancesEnergy', 'dft', 0.5)] -> correlation
dropout_values = [None, 0.5, 0.75, 0.85, 0.9, 0.95, 0.97, 0.99]



log_path_flaml = '/home/sim/Desktop/TS Extrinsic Regression/PredictionsAllDatasets/'
log_path_agluon = '/home/sim/Desktop/TS Extrinsic Regression/PredictionsAllDatasets_agluon/'

ds_names = [
    #'NewsTitleSentiment',
    'BenzeneConcentration',
    'BeijingPM25Quality',
    'IEEEPPG',
    'FloodModeling1',
    'HouseholdPowerConsumption1',
    'Covid3Month',
    'AppliancesEnergy',
    'BeijingPM10Quality'


]


tqs = ['dct', 'dft', 'dwt']


for log_path in [log_path_flaml, log_path_agluon]:
    correlations = {}

    for ds_name in ds_names:
        path_to_runs = log_path + f'{ds_name}/'

        features_raw = pd.read_parquet('/home/sim/Desktop/TS Extrinsic Regression/features_dfs/' + ds_name + f'/NONE_NONE_{ds_name}_features_TEST').drop('target', axis=1)
        #features_raw = load_and_prepare_everything(all_ds[ds_name], None, None)[1].drop('target', axis=1)
        
        path_to_pred = path_to_runs + f'NONE_{ds_name}_predictions.npy'
        prediction_raw = np.load(path_to_pred)
        features_raw['prediction'] = prediction_raw


        for tq in tqs:
            for dval in dropout_values[1:]:
                

                features = pd.read_parquet('/home/sim/Desktop/TS Extrinsic Regression/features_dfs/' + ds_name + f'/{dval}_{tq}_{ds_name}_features_TEST').drop('target', axis=1)
                #features = load_and_prepare_everything(all_ds[ds_name], tq, dval)[1].drop('target', axis=1)
                
                path_to_pred = path_to_runs + f'{dval}_{tq}_{ds_name}_predictions.npy'


                prediction = np.load(path_to_pred)

                features['prediction'] = prediction
                
                # Add Subtraction and Correlation
                sub_df = (features_raw - features)


                # Calculate the number of unique values in each column
                unique_counts = sub_df.nunique()
                # Filter out columns with only one unique value (zero variance), to avoid nan values in correlation colunm
                sub_df = sub_df.loc[:, unique_counts > 1]


                corrs = sub_df.corrwith(sub_df['prediction'])
                correlations[(ds_name,tq,dval)] = corrs

                print(f'{ds_name}, {tq}, {dval} correlation calculated')
                #print(corrs)

    if log_path_agluon == log_path:
        with open('correlations_agluon.pkl', 'wb') as file:
            pickle.dump(correlations, file)
    else:
        with open('correlations_flaml.pkl', 'wb') as file:
            pickle.dump(correlations, file)