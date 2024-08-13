import os, socket

if socket.gethostname() != "sim-IdeaPad-5-14ALC05":
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[var] = "40" 




###   TO RUN FLAML: UNCOMMENT FLAML_AND_FWIZTEST AND Uncomment CREATE_PREPARED_DATA

#from flaml_and_fwiztest_wlogs import run_flaml
from agluon import run_agluon



import argparse
from utils.personal_utils import *
#from create_prepared_data_tsfresh import *





path = '/home/simon/TSER/preparedData/'

path_ts = '/home/simon/TSER/Time_Series_Data/'


# For Agluon change all occurences
# For new Dataset, change ds_name
# Maybe add parameter option to do all test at once! -> delete before that the old experiments!

def tsf_or_f():

    ds_names = [
    'AppliancesEnergy',
    'NewsTitleSentiment',
    'BenzeneConcentration',
    'BeijingPM25Quality',
    'IEEEPPG',
    'FloodModeling1',
    'HouseholdPowerConsumption1',
    'Covid3Month'
    ]


    for ds_name in ds_names:

        for time in [15]:
            source_path_tsf = path + ds_name + '_TRAIN_None__ts_and_features.csv'
            source_path_f = path + ds_name + '_TRAIN_None__features.csv'

            run_flaml(source_path_tsf, 'Test if tsf or f', f'tsf {ds_name} Flaml {time}min', time * 60)
            run_flaml(source_path_f, 'Test if tsf or f', f'f {ds_name} Flaml {time}min', time * 60)


def best_time():
    # Letting run only with features

    ds_names = [
    'AppliancesEnergy',
    'NewsTitleSentiment',
    'BenzeneConcentration',
    'BeijingPM25Quality',
    'IEEEPPG',
    'FloodModeling1',
    'HouseholdPowerConsumption1',
    'Covid3Month'
    ]

    times = [1 ,5, 15, 20, 25, 30, 60]

    for ds_name in ds_names:

        source_path = path + ds_name + '_TRAIN_None__features.csv'


        for time in times:
            run_flaml(source_path, 'Test_best_runtime', f'{time}min_{ds_name}_Flaml_f', time * 60)




def test_deterministic():
    source_path = path + ds_name + '_TRAIN_None__features.csv'

    for i in range(5):
        run_flaml(source_path, 'Test deterministic', f'Run{i+1} Appliances Flaml 5min', 5 * 60)




# still with manually loading atm!
def run_flaml_all():

    # Appliances fully loaded
    # Newstitle only dct and half dft, other missing


    ds_names = [
    'NewsTitleSentiment',   
    'AppliancesEnergy',
    #'HouseholdPowerConsumption1',
    #'BenzeneConcentration',
    #'IEEEPPG',
    #'FloodModeling1',
    #'Covid3Month',
    #'BeijingPM25Quality',
    ]


    # Create folder to save the outputs
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    folder_path = os.path.join(parent_directory, 'PredictionsAllDatasets_agluon')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


    for ds_name in ds_names:
        #!! I add each solution and overwrite in case
        ds_path = os.path.join(folder_path, ds_name)
        if not os.path.exists(ds_path):
            os.makedirs(ds_path)


        data_train_path = path_ts + ds_name + "_TRAIN.ts"
        data_test_path = path_ts + ds_name + "_TEST.ts"


        # -> Can get data directly by loading it from the parquet file!
        train_data, train_features = load_and_prepare_everything(data_train_path, None, -1)
        test_data, test_features = load_and_prepare_everything(data_test_path, None, -1)



        y_prediction_none = run_flaml(f'{ds_name}_Runs', f'NONE_{ds_name}_20min_Flaml_f', 20 * 60, train_features, test_features)
        np.save(os.path.join(ds_path, f'NONE_{ds_name}_predictions.npy'), y_prediction_none)


        for tq in ['dct', 'dft', 'dwt']:
            for i in [0.5, 0.75, 0.85, 0.9, 0.95, 0.97, 0.99]:

                
                train_data, train_features = load_and_prepare_everything(data_train_path, tq, i)
                test_data, test_features = load_and_prepare_everything(data_test_path, tq, i)

                y_prediction = run_flaml(f'{ds_name}_Runs', f'{i}_{tq}_{ds_name}_20min_Flaml_f', 20 * 60, train_features, test_features)
                np.save(os.path.join(ds_path, f'{i}_{tq}_{ds_name}_predictions.npy'), y_prediction)




def run_agluon_all():
    ds_names = [
    #'NewsTitleSentiment',   
    'AppliancesEnergy',
    'HouseholdPowerConsumption1',
    'BenzeneConcentration',
    'IEEEPPG',
    'FloodModeling1',
    'Covid3Month',
    'BeijingPM25Quality',
    ]


    # Create folder to save the outputs
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    folder_path = os.path.join(parent_directory, 'PredictionsAllDatasets_agluon')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    

    for ds_name in ds_names:
            #!! I add each solution and overwrite in case
            ds_path = os.path.join(folder_path, ds_name)
            if not os.path.exists(ds_path):
                os.makedirs(ds_path)

            features_path = '/home/simon/TSER/features_dfs/' + ds_name + '/'
            #features_path = '/home/sim/Desktop/TS Extrinsic Regression/features_dfs/' + ds_name + '/'

            train_features = pd.read_parquet(features_path + f'NONE_NONE_{ds_name}_features_TRAIN')
            test_features = pd.read_parquet(features_path + f'NONE_NONE_{ds_name}_features_TEST')



            # print run names for nohup-out.log file
            print('SimonLog: ' + f'{ds_name}_NONE_NONE')
            y_prediction_none = run_agluon(f'{ds_name}_Runs', f'NONE_{ds_name}_20min_Agluon_f', 20 * 60 , train_features, test_features)

            np.save(os.path.join(ds_path, f'NONE_{ds_name}_predictions.npy'), y_prediction_none)



            for tq in ['dct', 'dft', 'dwt']:
                for i in [0.5, 0.75, 0.85, 0.9, 0.95, 0.97, 0.99]:
                    
                    train_features = pd.read_parquet(features_path + f'{i}_{tq}_{ds_name}_features_TRAIN')
                    test_features = pd.read_parquet(features_path + f'{i}_{tq}_{ds_name}_features_TEST')



                    if ds_name == 'NewsTitleSentiment':
                        data_train_path = path_ts + ds_name + "_TRAIN.ts"
                        data_test_path = path_ts + ds_name + "_TEST.ts"

                        # _ , train_features = load_and_prepare_everything(data_train_path, tq, i)
                        #_ , test_features = load_and_prepare_everything(data_test_path, tq, i)

                    print('SimonLog: ' + f'{ds_name}_{i}_{tq}')
                    y_prediction = run_agluon(f'{ds_name}_Runs', f'{i}_{tq}_{ds_name}_20min_Agluon_f', 20 * 60, train_features, test_features)
                    np.save(os.path.join(ds_path, f'{i}_{tq}_{ds_name}_predictions.npy'), y_prediction)










if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run specific function.')
    parser.add_argument('function', type=str, choices=['tsf_or_f', 'best_time', 'test_deterministic', 'run_flaml_all', 'run_agluon_all'],
                        help='The function to run: tsf_or_f, best_time, test_deterministic, or run_flaml')
    
    args = parser.parse_args()
    
    if args.function == 'tsf_or_f':
        tsf_or_f()
    elif args.function == 'best_time':
        best_time()
    elif args.function == 'test_deterministic':
        test_deterministic()
    elif args.function == 'run_flaml_all':
        run_flaml_all()
    elif args.function == 'run_agluon_all':
        run_agluon_all()
    












    
